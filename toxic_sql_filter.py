"""
ToxicSQLFilter: Combined Keyword + AST-Structural Filter
=========================================================
Implements the two-stage Output-level Control defense described in:

  ToxicSQL: Backdoor Attacks on Text-to-SQL via Toxic Query Injection
  (arXiv 2503.05445v2, Section 8 – Evaluation on Defense)

Pipeline:
  Stage 1  Keyword Gate  — fast O(n·rules) regex scan (KeywordFilter)
                           Immediately rejects/flags obvious malicious patterns:
                           OR '1'='1', --, SLEEP, DROP, INSERT, UNION, …

  Stage 2  AST Gate      — structural parse + skeleton analysis (ASTAlignDetector)
                           Catches structurally-anomalous queries that survive
                           the keyword gate (stealthy / obfuscated injections).

  Stage 3  Verdict       — fuses both signals into a single FilterResult with
                           an action: ALLOW | FLAG_FOR_REVIEW | REJECT

The filter is intentionally lightweight — it does NOT use embeddings or ML
models, making it suitable as a real-time guard in front of a Text-to-SQL
inference endpoint.

Usage
-----
    from toxic_sql_filter import ToxicSQLFilter

    f = ToxicSQLFilter()
    result = f.filter(sql="SELECT * FROM users WHERE id = 1 OR 1=1 --", nlq="Get user 1")
    print(result.action)          # → REJECT
    print(result.explain())       # human-readable report
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ast_safe_sql.detectors.keyword_filter import KeywordFilter, KeywordFilterResult
from ast_safe_sql.detectors.ast_align import ASTAlignDetector

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class FilterResult:
    """Verdict produced by :class:`ToxicSQLFilter`."""

    action: str
    """ALLOW | FLAG_FOR_REVIEW | REJECT"""

    is_toxic: bool
    """True when action is REJECT or FLAG_FOR_REVIEW."""

    overall_risk_score: float
    """0.0–1.0 fused risk score."""

    # Stage 1 — keyword
    keyword_flagged: bool = False
    keyword_risk_score: float = 0.0
    keyword_matched_rules: List[str] = field(default_factory=list)
    keyword_attack_types: List[str] = field(default_factory=list)
    keyword_recommendation: str = "ALLOW"

    # Stage 2 — AST structural
    ast_parsed: bool = False
    ast_alignment_score: float = 1.0
    ast_suspicious_patterns: List[str] = field(default_factory=list)
    ast_skeleton: Dict = field(default_factory=dict)

    # Explanation
    reasons: List[str] = field(default_factory=list)

    def explain(self) -> str:
        """Return a human-readable report of the filter decision."""
        icon = {"ALLOW": "✅", "FLAG_FOR_REVIEW": "⚠️", "REJECT": "🚨"}.get(
            self.action, "❓"
        )
        lines = [
            f"{icon} Action: {self.action}  (risk={self.overall_risk_score:.2f})",
            "",
        ]
        if self.reasons:
            lines.append("  Reasons:")
            for r in self.reasons:
                lines.append(f"    • {r}")
            lines.append("")

        if self.keyword_flagged:
            lines.append(
                f"  [Stage 1 – Keyword] risk={self.keyword_risk_score:.2f}, "
                f"rules={self.keyword_matched_rules}, "
                f"attack_types={self.keyword_attack_types}"
            )

        if not self.ast_parsed:
            lines.append("  [Stage 2 – AST] parse FAILED (query not valid SQL)")
        else:
            lines.append(
                f"  [Stage 2 – AST] alignment={self.ast_alignment_score:.2f}, "
                f"suspicious={self.ast_suspicious_patterns}"
            )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main filter class
# ---------------------------------------------------------------------------

class ToxicSQLFilter:
    """
    Two-stage keyword + AST filter against ToxicSQL backdoor attacks.

    Strategy (v2 — no keyword hard-reject):
    ----------------------------------------
    Keyword filter is a *score contributor only* — it raises the risk score
    but NEVER unilaterally rejects a query.  The AST structural analysis
    always runs and the fused score decides the verdict.

    This prevents false positives from legitimate SQL that happens to
    contain words like UNION, INSERT, or -- (e.g. comments in stored procs,
    UNION in reporting queries, etc.).

    Decision rule:
        overall_risk_score = KW_WEIGHT * kw_risk + AST_WEIGHT * ast_risk
        REJECT          → overall_risk_score ≥ reject_threshold
                          AND (keyword confirmed OR ast confirmed)
        FLAG_FOR_REVIEW → overall_risk_score ≥ flag_threshold
        ALLOW           → otherwise

    Parameters
    ----------
    ast_alignment_threshold:
        AST alignment scores below this contribute max AST risk.  Default: 0.6.
    ast_suspicious_weight:
        Risk added per AST suspicious pattern (no reference SQL).  Default: 0.20.
    flag_threshold:
        overall_risk_score at which action becomes FLAG_FOR_REVIEW.  Default: 0.35.
    reject_threshold:
        overall_risk_score at which action becomes REJECT (requires both
        keyword AND AST signals to be positive).  Default: 0.60.
    """

    # Weights for fusing keyword + AST signals — keyword is secondary
    _KW_WEIGHT: float = 0.40
    _AST_WEIGHT: float = 0.60

    def __init__(
        self,
        ast_alignment_threshold: float = 0.6,
        ast_suspicious_weight: float = 0.20,
        flag_threshold: float = 0.35,
        reject_threshold: float = 0.60,
    ) -> None:
        self.ast_alignment_threshold = ast_alignment_threshold
        self.ast_suspicious_weight = ast_suspicious_weight
        self.flag_threshold = flag_threshold
        self.reject_threshold = reject_threshold

        # KeywordFilter in non-strict mode: its recommendation is IGNORED —
        # we only use its risk score as a numeric signal.
        self._keyword_filter = KeywordFilter(strict_mode=False)
        self._ast_detector = ASTAlignDetector()

        logger.info(
            "ToxicSQLFilter ready | kw_weight=%.2f | ast_weight=%.2f | "
            "flag_threshold=%.2f | reject_threshold=%.2f | ast_threshold=%.2f",
            self._KW_WEIGHT, self._AST_WEIGHT,
            flag_threshold, reject_threshold, ast_alignment_threshold,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def filter(
        self,
        sql: str,
        nlq: str = "",
        expected_sql: Optional[str] = None,
    ) -> FilterResult:
        """
        Run the two-stage filter on *sql*.

        Parameters
        ----------
        sql:
            The SQL query generated by the Text-to-SQL model.
        nlq:
            Original natural-language question (used for context in logging).
        expected_sql:
            Optional clean reference SQL for structural comparison (Stage 2).
            When provided, tree similarity is computed against it.

        Returns
        -------
        FilterResult
            Full verdict with per-stage details.
        """
        reasons: List[str] = []

        # ── Stage 1: Keyword Gate ─────────────────────────────────────
        kw: KeywordFilterResult = self._keyword_filter.filter(sql)

        # ── Stage 2: AST Gate ─────────────────────────────────────────
        # IMPORTANT: run detect_suspicious_patterns on the RAW sql (not cleaned),
        # because _clean_sql() inside parse_sql_to_ast() strips injection artifacts
        # like OR 1=1, --, SLEEP() before parsing, causing patterns to be missed.
        ast_parsed = False
        ast_alignment_score = 1.0
        ast_skeleton: Dict = {}

        # Run pattern detection on raw SQL first
        susp = self._ast_detector.detect_suspicious_patterns(sql)
        ast_suspicious_patterns: List[str] = susp.get("patterns_found", [])

        ast_tree = self._ast_detector.parse_sql_to_ast(sql)
        if ast_tree is not None:
            ast_parsed = True
            ast_skeleton = self._ast_detector.extract_skeleton(ast_tree)

            # Structural comparison if reference SQL given
            if expected_sql:
                ref_tree = self._ast_detector.parse_sql_to_ast(expected_sql)
                if ref_tree is not None:
                    ast_alignment_score = self._ast_detector.compute_tree_similarity(
                        ast_tree, ref_tree
                    )
                    if ast_alignment_score < self.ast_alignment_threshold:
                        reasons.append(
                            f"AST alignment {ast_alignment_score:.2f} < "
                            f"threshold {self.ast_alignment_threshold:.2f}"
                        )

        else:
            # Parse failure — likely severely malformed / injected query
            ast_parsed = False
            ast_alignment_score = 0.0
            reasons.append("SQL could not be parsed — possibly malformed/injected")

        # ── Stage 3: Fuse signals ─────────────────────────────────────
        # keyword_risk_score is already normalized to [0,1] by KeywordFilter
        # (max matched severity / CRITICAL_weight), so 1 CRITICAL hit → 1.0.
        kw_risk = kw.keyword_risk_score

        # ast_risk: each suspicious pattern adds ast_suspicious_weight.
        # Parse failure = maximum structural risk (1.0).
        if not ast_parsed:
            ast_risk = 1.0
        elif ast_suspicious_patterns:
            ast_risk = min(1.0, len(ast_suspicious_patterns) * self.ast_suspicious_weight)
        else:
            ast_risk = 0.0   # clean structure

        overall_risk_score = min(
            1.0,
            self._KW_WEIGHT * kw_risk + self._AST_WEIGHT * ast_risk
        )

        action, is_toxic = self._decide_action(
            kw_result=kw,
            ast_parsed=ast_parsed,
            ast_alignment_score=ast_alignment_score,
            ast_suspicious_patterns=ast_suspicious_patterns,
            overall_risk_score=overall_risk_score,
        )

        # Collect reasons
        if kw.flagged:
            reasons.append(
                f"Keyword gate: {', '.join(kw.matched_rules)} → {kw.recommendation}"
            )
        if ast_suspicious_patterns:
            reasons.append(f"AST structural anomalies: {ast_suspicious_patterns}")

        result = FilterResult(
            action=action,
            is_toxic=is_toxic,
            overall_risk_score=overall_risk_score,
            keyword_flagged=kw.flagged,
            keyword_risk_score=kw.keyword_risk_score,
            keyword_matched_rules=kw.matched_rules,
            keyword_attack_types=kw.attack_types_found,
            keyword_recommendation=kw.recommendation,
            ast_parsed=ast_parsed,
            ast_alignment_score=ast_alignment_score,
            ast_suspicious_patterns=ast_suspicious_patterns,
            ast_skeleton=ast_skeleton,
            reasons=reasons,
        )

        logger.info(
            "ToxicSQLFilter | action=%s | risk=%.2f | kw_flagged=%s | "
            "ast_align=%.2f | sql=%.60s",
            action, overall_risk_score, kw.flagged, ast_alignment_score, sql,
        )

        return result

    def batch_filter(
        self,
        queries: List[Dict],
        sql_key: str = "sql",
        nlq_key: str = "nlq",
    ) -> List[FilterResult]:
        """
        Filter a batch of queries.

        Parameters
        ----------
        queries:
            List of dicts, each with at least *sql_key* field.
        sql_key / nlq_key:
            Dict keys for SQL and NLQ strings.

        Returns
        -------
        List[FilterResult], one per input query.
        """
        results = []
        for q in queries:
            sql = q.get(sql_key, "")
            nlq = q.get(nlq_key, "")
            results.append(self.filter(sql=sql, nlq=nlq))
        return results

    # ------------------------------------------------------------------
    # Decision logic
    # ------------------------------------------------------------------

    def _decide_action(
        self,
        kw_result: KeywordFilterResult,
        ast_parsed: bool,
        ast_alignment_score: float,
        ast_suspicious_patterns: List[str],
        overall_risk_score: float,
    ) -> tuple[str, bool]:
        """
        Return (action, is_toxic) based purely on the fused risk score.

        Decision table (v2 — no keyword hard-reject)
        ---------------------------------------------
        Keyword is a score contributor ONLY — it never triggers REJECT alone.
        AST must confirm before a query is rejected.

        overall_risk_score ≥ reject_threshold
            AND (keyword flagged OR AST found suspicious patterns)  → REJECT
        overall_risk_score ≥ flag_threshold                         → FLAG_FOR_REVIEW
        Otherwise                                                   → ALLOW

        Parse failure is treated as high AST risk (ast_alignment_score=0.0)
        so it naturally pushes overall_risk_score up, but still needs
        keyword confirmation to reach REJECT.
        """
        kw_confirmed = kw_result.flagged
        ast_confirmed = (not ast_parsed) or bool(ast_suspicious_patterns) or \
                        (ast_alignment_score < self.ast_alignment_threshold)

        if overall_risk_score >= self.reject_threshold and (kw_confirmed or ast_confirmed):
            return "REJECT", True

        if overall_risk_score >= self.flag_threshold:
            return "FLAG_FOR_REVIEW", True

        return "ALLOW", False
