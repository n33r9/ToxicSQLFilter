"""
AST-Align Detector: Structure-Aware Anomaly Detection
Parses SQL to AST, extracts skeleton, and compares structural consistency
"""

from typing import Dict, Optional, Tuple, List
import logging
import re
import sqlglot
from sqlglot.expressions import Expression, Select, Where, From
import networkx as nx

logger = logging.getLogger(__name__)


class ASTAlignDetector:
    """Detects anomalies via AST structure alignment"""

    def __init__(self):
        """Initialize AST alignment detector"""
        self.parser = sqlglot
        logger.info("AST-Align detector initialized")

    def _clean_sql(self, sql: str) -> str:
        """Basic sanitization to remove common injection artifacts that break parsing.

        - Removes trailing injection clauses like `OR 1 = 1` or `OR 'a'='a'`.
        - Removes stray `OR` after semicolons or before EOF.
        - Collapses multiple whitespace.
        """
        if not sql:
            return sql
        s = sql
        # Normalize whitespace
        s = re.sub(r"\s+", " ", s).strip()

        # Remove obvious injection tautologies like OR 1=1 at the end
        s = re.sub(r";?\s*OR\s+1\s*=\s*1\s*;?$", "", s, flags=re.IGNORECASE)
        s = re.sub(r";?\s*OR\s+'[^']+'\s*=\s*'[^']+'\s*;?$", "", s, flags=re.IGNORECASE)

        # Convert double-quoted string literals to single quotes safely
        def _dq_to_sq(match):
            inner = match.group(1)
            inner = inner.replace("'", "''")
            return "'{}'".format(inner)

        s = re.sub(r'"([^\"]*)"', _dq_to_sq, s)

        # Escape single quotes inside existing single-quoted literals by doubling them
        def _escape_sq(match):
            inner = match.group(1)
            inner = inner.replace("'", "''")
            return "'{}'".format(inner)

        s = re.sub(r"'([^']*)'", _escape_sq, s)

        # Normalize set operators spacing
        s = re.sub(r"\bINTERSECT\b", "INTERSECT", s, flags=re.IGNORECASE)
        s = re.sub(r"\bUNION\b", "UNION", s, flags=re.IGNORECASE)

        # Remove dangling OR tokens
        s = re.sub(r"\bOR\s*$", "", s, flags=re.IGNORECASE)
        s = re.sub(r";\s*OR\s+", "; ", s, flags=re.IGNORECASE)

        # Attempt simple balancing for parentheses: if too many closing parens, trim trailing content
        open_paren = s.count('(')
        close_paren = s.count(')')
        if close_paren > open_paren:
            # Trim from the end until balanced or no change
            while close_paren > open_paren and ')' in s:
                last_close = s.rfind(')')
                if last_close == -1:
                    break
                s = s[:last_close]
                open_paren = s.count('(')
                close_paren = s.count(')')
            s = s.strip()

        # Final whitespace collapse
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def parse_sql_to_ast(self, sql_query: str) -> Optional[Expression]:
        """
        Parse SQL query to Abstract Syntax Tree

        Args:
            sql_query: SQL query string

        Returns:
            sqlglot Expression (AST) or None if parsing fails
        """
        cleaned = self._clean_sql(sql_query)
        try:
            tree = self.parser.parse_one(cleaned, read="postgres")
            return tree
        except Exception as e:
            logger.error(f"Failed to parse SQL: {cleaned[:50]}... Error: {e}")
            return None

    def extract_skeleton(self, ast: Expression) -> Dict:
        """
        Extract SQL skeleton (abstract structure)

        Args:
            ast: Abstract Syntax Tree

        Returns:
            Dictionary containing skeleton information
        """
        skeleton = {
            "query_type": self._get_query_type(ast),
            "main_clauses": [],
            "has_where": False,
            "has_subquery": False,
            "has_union": False,
            "has_join": False,
            "has_aggregate": False,
            "table_count": 0,
            "column_count": 0,
            "condition_complexity": 0,
            "clause_order": []
        }

        # Extract main structure
        if isinstance(ast, Select):
            # Check for WHERE clause
            if ast.find(Where):
                skeleton["has_where"] = True
                skeleton["clause_order"].append("WHERE")

            # Check for subqueries
            if ast.find(Select, bfs=False) and ast != ast.find(Select, bfs=False):
                skeleton["has_subquery"] = True

            # Check for UNION/EXCEPT/INTERSECT
            if ast.find(sqlglot.exp.Union):
                skeleton["has_union"] = True

            # Check for JOINs
            from_clause = ast.find(From)
            if from_clause:
                skeleton["has_join"] = self._has_join(from_clause)
                skeleton["table_count"] = self._count_tables(from_clause)

            # Check for aggregates
            if ast.find(sqlglot.exp.AggFunc):
                skeleton["has_aggregate"] = True

            # Count columns
            skeleton["column_count"] = len(ast.expressions)

            # Estimate condition complexity
            if skeleton["has_where"]:
                where_clause = ast.find(Where)
                skeleton["condition_complexity"] = self._estimate_complexity(where_clause)

        skeleton["clause_order"].sort()
        return skeleton

    def _get_query_type(self, ast: Expression) -> str:
        """Determine query type (SELECT, INSERT, UPDATE, DELETE, etc.)"""
        if isinstance(ast, sqlglot.exp.Select):
            return "SELECT"
        elif isinstance(ast, sqlglot.exp.Insert):
            return "INSERT"
        elif isinstance(ast, sqlglot.exp.Update):
            return "UPDATE"
        elif isinstance(ast, sqlglot.exp.Delete):
            return "DELETE"
        else:
            return type(ast).__name__

    def _has_join(self, from_clause: Expression) -> bool:
        """Check if FROM clause contains JOIN"""
        return bool(from_clause.find(sqlglot.exp.Join))

    def _count_tables(self, from_clause: Expression) -> int:
        """Count number of tables in FROM clause"""
        tables = list(from_clause.find_all(sqlglot.exp.Table))
        return len(tables)

    def _estimate_complexity(self, where_clause: Expression) -> int:
        """Estimate WHERE clause complexity (number of conditions)"""
        and_conditions = len(list(where_clause.find_all(sqlglot.exp.And)))
        or_conditions = len(list(where_clause.find_all(sqlglot.exp.Or)))
        not_conditions = len(list(where_clause.find_all(sqlglot.exp.Not)))
        return and_conditions + or_conditions + not_conditions

    def compute_tree_similarity(
        self,
        ast1: Expression,
        ast2: Expression
    ) -> float:
        """
        Compute AST tree similarity using structural features

        Args:
            ast1: First AST
            ast2: Second AST

        Returns:
            Similarity score (0-1), where 1 is identical
        """
        # Extract structural features
        feat1 = self._extract_tree_features(ast1)
        feat2 = self._extract_tree_features(ast2)

        # Compare features
        total_features = len(feat1)
        if total_features == 0:
            return 0.0

        matching = sum(1 for k in feat1 if k in feat2 and feat1[k] == feat2[k])
        similarity = matching / total_features

        return similarity

    def _extract_tree_features(self, ast: Expression) -> Dict:
        """Extract structural features from AST"""
        features = {}

        # Count different node types
        features["node_count"] = len(list(ast.find_all(Expression)))
        features["select_count"] = len(list(ast.find_all(Select)))
        features["where_count"] = len(list(ast.find_all(Where)))
        features["join_count"] = len(list(ast.find_all(sqlglot.exp.Join)))
        features["aggregate_count"] = len(list(ast.find_all(sqlglot.exp.AggFunc)))
        features["subquery_count"] = len(list(ast.find_all(sqlglot.exp.Subquery)))
        features["union_count"] = len(list(ast.find_all(sqlglot.exp.Union)))
        features["table_count"] = len(list(ast.find_all(sqlglot.exp.Table)))

        # Query type
        features["query_type"] = self._get_query_type(ast)

        return features

    def detect_suspicious_patterns(self, sql_query: str) -> Dict:
        """
        Detect suspicious SQL patterns associated with backdoors

        Args:
            sql_query: SQL query string

        Returns:
            Dictionary of suspicious pattern detection
        """
        suspicious = {
            "has_comment_injection": False,
            "has_tautology": False,
            "has_time_delay": False,
            "has_piggy_back": False,
            "has_union_select": False,
            "patterns_found": []
        }

        query_lower = sql_query.lower()

        # Comment patterns (Comment injection backdoor)
        if "--" in query_lower or "/*" in query_lower:
            suspicious["has_comment_injection"] = True
            suspicious["patterns_found"].append("SQL Comment detected")

        # Tautology patterns (always true conditions)
        if "1=1" in query_lower or "'1'='1'" in query_lower or "true" in query_lower:
            suspicious["has_tautology"] = True
            suspicious["patterns_found"].append("Tautology condition detected")

        # Time delay patterns (Delay backdoor)
        if any(x in query_lower for x in ["sleep", "delay", "pg_sleep", "waitfor"]):
            suspicious["has_time_delay"] = True
            suspicious["patterns_found"].append("Time delay function detected")

        # Piggy-back patterns (extra queries)
        if query_lower.count(";") > 1:
            suspicious["has_piggy_back"] = True
            suspicious["patterns_found"].append("Multiple statements detected")

        # UNION injection — only flag when paired with injection markers (NULL padding or comment)
        # Plain "UNION SELECT" is valid SQL and causes false positives on reporting queries
        has_union_inject = (
            "union" in query_lower and "select" in query_lower and
            ("null" in query_lower or "--" in query_lower or "information_schema" in query_lower)
        )
        if has_union_inject:
            suspicious["has_union_select"] = True
            suspicious["patterns_found"].append("UNION SELECT injection pattern detected")

        return suspicious

    def check_alignment(
        self,
        generated_sql: str,
        natural_language_query: str,
        schema_description: str,
        expected_skeleton: Optional[str] = None
    ) -> Dict:
        """
        Main alignment checking function

        Args:
            generated_sql: Generated SQL query
            natural_language_query: Original NLQ
            schema_description: Database schema
            expected_skeleton: Optional expected SQL skeleton

        Returns:
            Dictionary with alignment scores and details
        """
        result = {
            "alignment_score": 0.0,
            "tree_similarity": 0.0,
            "suspicious_patterns": {},
            "skeleton_analysis": {},
            "anomaly_indicators": [],
            "details": {}
        }

        # Parse generated SQL
        ast_generated = self.parse_sql_to_ast(generated_sql)
        if not ast_generated:
            result["alignment_score"] = 0.0
            result["anomaly_indicators"].append("Failed to parse generated SQL")
            return result

        # Extract skeleton
        skeleton_generated = self.extract_skeleton(ast_generated)
        result["skeleton_analysis"]["generated"] = skeleton_generated

        # Detect suspicious patterns
        suspicious = self.detect_suspicious_patterns(generated_sql)
        result["suspicious_patterns"] = suspicious

        if suspicious["patterns_found"]:
            result["anomaly_indicators"].extend(suspicious["patterns_found"])

        # If expected skeleton provided, compare
        tree_similarity = 1.0
        if expected_skeleton:
            ast_expected = self.parse_sql_to_ast(expected_skeleton)
            if ast_expected:
                tree_similarity = self.compute_tree_similarity(ast_generated, ast_expected)
                skeleton_expected = self.extract_skeleton(ast_expected)
                result["skeleton_analysis"]["expected"] = skeleton_expected

                # Check skeleton consistency
                consistency = self._check_skeleton_consistency(
                    skeleton_generated,
                    skeleton_expected
                )
                result["details"]["skeleton_consistency"] = consistency

        result["tree_similarity"] = tree_similarity

        # Compute alignment score
        # Score decreases if suspicious patterns found
        pattern_penalty = len(suspicious["patterns_found"]) * 0.15
        alignment_score = max(0.0, tree_similarity - pattern_penalty)

        result["alignment_score"] = alignment_score

        logger.info(f"AST-Align check: similarity={tree_similarity:.2f}, "
                    f"alignment={alignment_score:.2f}, "
                    f"suspicious_patterns={len(suspicious['patterns_found'])}")

        return result

    def _check_skeleton_consistency(
        self,
        generated: Dict,
        expected: Dict
    ) -> Dict:
        """Check consistency between generated and expected skeletons"""
        consistency = {
            "query_type_match": generated["query_type"] == expected["query_type"],
            "where_clause_match": generated["has_where"] == expected["has_where"],
            "join_match": generated["has_join"] == expected["has_join"],
            "aggregate_match": generated["has_aggregate"] == expected["has_aggregate"],
            "table_count_match": generated["table_count"] == expected["table_count"],
            "inconsistencies": []
        }

        if not consistency["query_type_match"]:
            consistency["inconsistencies"].append("Query type mismatch")

        if not consistency["where_clause_match"]:
            consistency["inconsistencies"].append("WHERE clause mismatch")

        if not consistency["join_match"]:
            consistency["inconsistencies"].append("JOIN clause mismatch")

        consistency["consistency_score"] = sum([
            consistency["query_type_match"],
            consistency["where_clause_match"],
            consistency["join_match"],
            consistency["aggregate_match"],
            consistency["table_count_match"]
        ]) / 5

        return consistency
