"""
Policy Boundary Checker: Rule-based policy enforcement
Detects SQL patterns that violate security policies
"""

from typing import Dict, List, Set, Optional
import logging
import re

logger = logging.getLogger(__name__)


class PolicyBoundaryChecker:
    """Enforces SQL security policies and detects boundary violations"""

    def __init__(self, strict_mode: bool = True):
        """
        Initialize policy checker

        Args:
            strict_mode: Enable strict policy checking
        """
        self.strict_mode = strict_mode
        self.violation_rules = self._initialize_rules()
        logger.info(f"Policy checker initialized (strict_mode={strict_mode})")

    def _initialize_rules(self) -> Dict:
        """Initialize policy violation rules"""
        return {
            "forbidden_keywords": {
                "strict": [
                    "DROP", "TRUNCATE", "ALTER TABLE", "CREATE TABLE",
                    "DELETE FROM", "INSERT INTO", "UPDATE",
                    "EXEC", "EXECUTE", "XP_", "SP_",
                    "LOAD_FILE", "INTO OUTFILE", "INTO DUMPFILE",
                    "SLEEP", "WAITFOR", "DELAY", "PG_SLEEP",
                    "SHUTDOWN", "REVOKE", "GRANT"
                ],
                "lenient": [
                    "DROP", "TRUNCATE", "DELETE FROM", "INSERT INTO"
                ]
            },
            "dangerous_functions": {
                "strict": [
                    "xp_cmdshell", "sp_executesql", "exec", "execute",
                    "system", "proc_open", "shell_exec", "mysql_query",
                    "load_file", "benchmark", "sleep", "pg_sleep",
                    "waitfor", "sys_exec", "cmd_shell", "sql_variant_property"
                ],
                "lenient": [
                    "xp_cmdshell", "sp_executesql", "exec", "execute"
                ]
            },
            "suspicious_patterns": {
                "comment_injection": r"(--\s*|/\*.*?\*/|#\s*)",
                "tautology": r"(1\s*=\s*1|'1'\s*=\s*'1'|true\s*=\s*true)",
                # UNION injection: only flag when combined with NULL padding or comment
                # (plain UNION SELECT is valid SQL in reporting queries)
                "union_inject": r"(?:union\s+(?:all\s+)?select\s+null|union\s+(?:all\s+)?select\b.{0,120}information_schema)",
                "blind_injection": r"(and\s+1\s*=\s*1|and\s+'1'\s*=\s*'1')",
                "time_delay": r"(?:sleep\s*\(|waitfor\s+delay|pg_sleep\s*\(|benchmark\s*\()",
                "stacked_queries": r";\s*(?:select|insert|update|delete|create|drop|alter)",
                "cartesian_product": r"cross\s+join",
                "subquery_bomb": r"select.*?select.*?select.*?select",
            }
        }

    def check_boundaries(self, sql_query: str) -> Dict:
        """
        Main boundary checking function

        Args:
            sql_query: SQL query string

        Returns:
            Dictionary with violation detection results
        """
        result = {
            "violation_detected": False,
            "violations": [],
            "risk_score": 0.0,
            "details": {},
            "policy_checks": {}
        }

        query_upper = sql_query.upper()
        query_lower = sql_query.lower()

        # Check forbidden keywords
        keyword_violations = self._check_forbidden_keywords(query_upper)
        result["policy_checks"]["forbidden_keywords"] = keyword_violations

        if keyword_violations:
            result["violations"].extend(keyword_violations)
            result["risk_score"] += 0.25

        # Check dangerous functions
        function_violations = self._check_dangerous_functions(query_lower)
        result["policy_checks"]["dangerous_functions"] = function_violations

        if function_violations:
            result["violations"].extend(function_violations)
            result["risk_score"] += 0.25

        # Check suspicious patterns
        pattern_violations = self._check_suspicious_patterns(query_lower)
        result["policy_checks"]["suspicious_patterns"] = pattern_violations

        if pattern_violations:
            result["violations"].extend(pattern_violations)
            result["risk_score"] += 0.30

        # Check structural anomalies
        structural_violations = self._check_structural_anomalies(sql_query)
        result["policy_checks"]["structural_anomalies"] = structural_violations

        if structural_violations:
            result["violations"].extend(structural_violations)
            result["risk_score"] += 0.20

        # Determine if violation detected
        result["violation_detected"] = len(result["violations"]) > 0

        # Cap risk score at 1.0
        result["risk_score"] = min(1.0, result["risk_score"])

        logger.info(f"Policy check: violations={len(result['violations'])}, "
                    f"risk_score={result['risk_score']:.2f}")

        return result

    def _check_forbidden_keywords(self, query_upper: str) -> List[str]:
        """Check for forbidden keywords"""
        violations = []
        keyword_set = (
            self.violation_rules["forbidden_keywords"]["strict"]
            if self.strict_mode
            else self.violation_rules["forbidden_keywords"]["lenient"]
        )

        for keyword in keyword_set:
            # Use word boundary regex to avoid false positives
            pattern = rf"\b{keyword}\b"
            if re.search(pattern, query_upper):
                violations.append(f"Forbidden keyword detected: {keyword}")

        return violations

    def _check_dangerous_functions(self, query_lower: str) -> List[str]:
        """Check for dangerous function calls"""
        violations = []
        function_set = (
            self.violation_rules["dangerous_functions"]["strict"]
            if self.strict_mode
            else self.violation_rules["dangerous_functions"]["lenient"]
        )

        for func in function_set:
            # Match function name followed by parenthesis or space
            pattern = rf"\b{func}\s*\("
            if re.search(pattern, query_lower):
                violations.append(f"Dangerous function detected: {func}")

        return violations

    def _check_suspicious_patterns(self, query_lower: str) -> List[str]:
        """Check for suspicious SQL injection patterns"""
        violations = []
        patterns = self.violation_rules["suspicious_patterns"]

        # Check each pattern type
        pattern_hits = {}

        for pattern_name, regex in patterns.items():
            if re.search(regex, query_lower, re.IGNORECASE):
                violations.append(f"Suspicious pattern detected: {pattern_name}")
                pattern_hits[pattern_name] = True

        return violations

    def _check_structural_anomalies(self, sql_query: str) -> List[str]:
        """Check for structural SQL anomalies"""
        violations = []

        # Check for multiple statements
        statement_count = sql_query.count(";")
        if statement_count > 1:
            violations.append("Multiple statements detected")

        # Check for excessive parentheses (possible subquery bomb)
        paren_count = sql_query.count("(")
        if paren_count > 10:
            violations.append("Excessive nesting detected (possible subquery bomb)")

        # Check for extremely long WHERE clause (>2000 chars)
        import re
        where_match = re.search(r"WHERE\s+(.+?)(?:GROUP BY|ORDER BY|LIMIT|HAVING|$)", 
                               sql_query, re.IGNORECASE | re.DOTALL)
        if where_match and len(where_match.group(1)) > 2000:
            violations.append("Abnormally long WHERE clause")

        # Check for unbalanced quotes
        single_quotes = sql_query.count("'")
        if single_quotes % 2 != 0:
            violations.append("Unbalanced quotes detected")

        return violations

    def get_policy_risk_level(self, risk_score: float) -> str:
        """
        Categorize risk level based on risk score

        Args:
            risk_score: Risk score (0-1)

        Returns:
            Risk level string: 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
        """
        if risk_score < 0.2:
            return "LOW"
        elif risk_score < 0.5:
            return "MEDIUM"
        elif risk_score < 0.8:
            return "HIGH"
        else:
            return "CRITICAL"

    def generate_policy_report(
        self,
        sql_query: str,
        result: Optional[Dict] = None
    ) -> Dict:
        """
        Generate comprehensive policy violation report

        Args:
            sql_query: SQL query
            result: Optional pre-computed check result

        Returns:
            Detailed policy report
        """
        if result is None:
            result = self.check_boundaries(sql_query)

        report = {
            "sql_query": sql_query[:100] + "..." if len(sql_query) > 100 else sql_query,
            "violations_count": len(result["violations"]),
            "risk_score": result["risk_score"],
            "risk_level": self.get_policy_risk_level(result["risk_score"]),
            "violations": result["violations"],
            "recommendations": self._get_recommendations(result),
            "timestamp": self._get_timestamp()
        }

        return report

    def _get_recommendations(self, result: Dict) -> List[str]:
        """Generate recommendations based on violations"""
        recommendations = []

        if len(result["violations"]) > 5:
            recommendations.append("Query contains multiple policy violations - REJECT")
        elif len(result["violations"]) > 0:
            recommendations.append("Query contains policy violations - ROUTE TO SANDBOX")

        if result["risk_score"] > 0.7:
            recommendations.append("High-risk query - Consider additional review")

        if any("DELETE" in v or "DROP" in v for v in result["violations"]):
            recommendations.append("Data modification/destruction detected - REJECT")

        if any("time_delay" in v.lower() for v in result["violations"]):
            recommendations.append("Time-based attack pattern detected - REJECT")

        if len(recommendations) == 0:
            recommendations.append("Query passes policy checks")

        return recommendations

    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()

    def enforce_policy(
        self,
        sql_query: str,
        policy_level: str = "medium"
    ) -> Dict:
        """
        Enforce policy based on configured level

        Args:
            sql_query: SQL query
            policy_level: 'lenient', 'medium', 'strict'

        Returns:
            Enforcement decision
        """
        # Adjust strictness
        old_strict = self.strict_mode
        self.strict_mode = (policy_level == "strict")

        result = self.check_boundaries(sql_query)

        self.strict_mode = old_strict

        decision = {
            "allowed": not result["violation_detected"],
            "policy_level": policy_level,
            "violations": result["violations"],
            "risk_score": result["risk_score"],
            "action": self._get_enforcement_action(result["violation_detected"], result["risk_score"])
        }

        return decision

    def _get_enforcement_action(
        self,
        violation_detected: bool,
        risk_score: float
    ) -> str:
        """Determine enforcement action"""
        if not violation_detected:
            return "ALLOW"
        elif risk_score < 0.3:
            return "ALLOW_WITH_LOGGING"
        elif risk_score < 0.6:
            return "SANDBOX"
        else:
            return "REJECT"
