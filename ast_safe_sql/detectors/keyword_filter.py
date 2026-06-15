"""
Keyword-Based Malicious Content Filter
---------------------------------------
Implements the Output-level Control (SQL Filtering) defense described in:

  ToxicSQL: Backdoor Attacks on Text-to-SQL via Toxic Query Injection
  (arXiv 2503.05445v2, Section 8 – Evaluation on Defense)

The paper recommends flagging generated SQL queries that contain potentially
harmful keywords/patterns:
  • OR '1'='1'  (tautology)
  • --           (end-of-line comment injection)
  • SLEEP        (time-delay attack)
  • DROP         (piggy-back / destructive DDL)
  • INSERT       (piggy-back data-manipulation)
  • UNION        (UNION-based injection)

This module extends the paper's keyword list with additional patterns
covering all four ToxicSQL attack types:
  1. End-of-Line Comment  →  comment_injection rules
  2. Delay                →  time_delay rules
  3. Piggy-Back Query     →  stacked_query / destructive_ddl rules
  4. Tautology            →  tautology rules

All rules are organised into named groups so callers can understand *why*
a query was flagged, not just *that* it was flagged.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rule definitions
# ---------------------------------------------------------------------------

@dataclass
class KeywordRule:
    """A single keyword/pattern rule."""
    name: str
    pattern: re.Pattern
    attack_type: str          # ToxicSQL attack category
    severity: str             # LOW | MEDIUM | HIGH | CRITICAL
    description: str
    weight: float = 1.0       # contribution to keyword_risk_score


# Severity → numeric weight mapping (used for risk score computation)
_SEVERITY_WEIGHT: Dict[str, float] = {
    "LOW": 0.15,
    "MEDIUM": 0.30,
    "HIGH": 0.55,
    "CRITICAL": 0.80,
}

# ---------------------------------------------------------------------------
# Rule catalogue (compiled once at import time)
# ---------------------------------------------------------------------------
_RULES: List[KeywordRule] = [

    # ── Tautology ────────────────────────────────────────────────────────
    KeywordRule(
        name="tautology_or_1_eq_1",
        pattern=re.compile(r"\bOR\s+1\s*=\s*1\b", re.IGNORECASE),
        attack_type="tautology",
        severity="CRITICAL",
        description="Classic OR 1=1 tautology – always-true condition injection",
        weight=_SEVERITY_WEIGHT["CRITICAL"],
    ),
    KeywordRule(
        name="tautology_quoted_string",
        pattern=re.compile(r"\bOR\s+'[^']*'\s*=\s*'[^']*'\b", re.IGNORECASE),
        attack_type="tautology",
        severity="CRITICAL",
        description="Quoted-string tautology e.g. OR 'a'='a'",
        weight=_SEVERITY_WEIGHT["CRITICAL"],
    ),
    KeywordRule(
        name="tautology_true_literal",
        pattern=re.compile(r"\bOR\s+TRUE\b", re.IGNORECASE),
        attack_type="tautology",
        severity="HIGH",
        description="OR TRUE tautology",
        weight=_SEVERITY_WEIGHT["HIGH"],
    ),
    KeywordRule(
        name="tautology_and_1_eq_1",
        pattern=re.compile(r"\bAND\s+1\s*=\s*1\b", re.IGNORECASE),
        attack_type="tautology",
        severity="HIGH",
        description="AND 1=1 blind-injection tautology",
        weight=_SEVERITY_WEIGHT["HIGH"],
    ),
    KeywordRule(
        name="tautology_quoted_and",
        pattern=re.compile(r"\bAND\s+'[^']*'\s*=\s*'[^']*'\b", re.IGNORECASE),
        attack_type="tautology",
        severity="HIGH",
        description="Quoted-string blind-injection tautology",
        weight=_SEVERITY_WEIGHT["HIGH"],
    ),

    # ── Comment injection ────────────────────────────────────────────────
    KeywordRule(
        name="comment_double_dash",
        pattern=re.compile(r"--"),
        attack_type="comment_injection",
        severity="HIGH",
        description="SQL end-of-line comment (--) – used to nullify trailing WHERE clauses",
        weight=_SEVERITY_WEIGHT["HIGH"],
    ),
    KeywordRule(
        name="comment_block",
        pattern=re.compile(r"/\*.*?\*/", re.DOTALL),
        attack_type="comment_injection",
        severity="HIGH",
        description="SQL block comment (/* ... */) – can hide injected payload",
        weight=_SEVERITY_WEIGHT["HIGH"],
    ),
    KeywordRule(
        name="comment_hash",
        pattern=re.compile(r"#\s"),
        attack_type="comment_injection",
        severity="MEDIUM",
        description="MySQL hash comment (#) – used to terminate query early",
        weight=_SEVERITY_WEIGHT["MEDIUM"],
    ),

    # ── Time-delay (Delay backdoor) ──────────────────────────────────────
    KeywordRule(
        name="delay_sleep",
        pattern=re.compile(r"\bSLEEP\s*\(", re.IGNORECASE),
        attack_type="time_delay",
        severity="CRITICAL",
        description="SLEEP() function – time-based blind injection / delay backdoor",
        weight=_SEVERITY_WEIGHT["CRITICAL"],
    ),
    KeywordRule(
        name="delay_pg_sleep",
        pattern=re.compile(r"\bPG_SLEEP\s*\(", re.IGNORECASE),
        attack_type="time_delay",
        severity="CRITICAL",
        description="PostgreSQL pg_sleep() – time-based delay",
        weight=_SEVERITY_WEIGHT["CRITICAL"],
    ),
    KeywordRule(
        name="delay_waitfor",
        pattern=re.compile(r"\bWAITFOR\s+DELAY\b", re.IGNORECASE),
        attack_type="time_delay",
        severity="CRITICAL",
        description="MSSQL WAITFOR DELAY – time-based delay",
        weight=_SEVERITY_WEIGHT["CRITICAL"],
    ),
    KeywordRule(
        name="delay_benchmark",
        pattern=re.compile(r"\bBENCHMARK\s*\(", re.IGNORECASE),
        attack_type="time_delay",
        severity="HIGH",
        description="MySQL BENCHMARK() – CPU-based timing attack",
        weight=_SEVERITY_WEIGHT["HIGH"],
    ),

    # ── Piggy-back / stacked queries ─────────────────────────────────────
    KeywordRule(
        name="stacked_select",
        pattern=re.compile(r";\s*SELECT\b", re.IGNORECASE),
        attack_type="piggy_back",
        severity="CRITICAL",
        description="Stacked SELECT after semicolon – piggy-back query",
        weight=_SEVERITY_WEIGHT["CRITICAL"],
    ),
    KeywordRule(
        name="stacked_drop",
        pattern=re.compile(r";\s*DROP\b", re.IGNORECASE),
        attack_type="piggy_back",
        severity="CRITICAL",
        description="DROP TABLE/DATABASE after semicolon – destructive piggy-back",
        weight=_SEVERITY_WEIGHT["CRITICAL"],
    ),
    KeywordRule(
        name="stacked_insert",
        pattern=re.compile(r";\s*INSERT\b", re.IGNORECASE),
        attack_type="piggy_back",
        severity="CRITICAL",
        description="INSERT after semicolon – data-manipulation piggy-back",
        weight=_SEVERITY_WEIGHT["CRITICAL"],
    ),
    KeywordRule(
        name="stacked_update",
        pattern=re.compile(r";\s*UPDATE\b", re.IGNORECASE),
        attack_type="piggy_back",
        severity="CRITICAL",
        description="UPDATE after semicolon – data-manipulation piggy-back",
        weight=_SEVERITY_WEIGHT["CRITICAL"],
    ),
    KeywordRule(
        name="stacked_delete",
        pattern=re.compile(r";\s*DELETE\b", re.IGNORECASE),
        attack_type="piggy_back",
        severity="CRITICAL",
        description="DELETE after semicolon – data-destruction piggy-back",
        weight=_SEVERITY_WEIGHT["CRITICAL"],
    ),
    KeywordRule(
        name="bare_drop",
        pattern=re.compile(r"\bDROP\s+(TABLE|DATABASE|SCHEMA|INDEX|VIEW|PROCEDURE|FUNCTION)\b",
                           re.IGNORECASE),
        attack_type="piggy_back",
        severity="CRITICAL",
        description="Bare DROP statement – destructive DDL",
        weight=_SEVERITY_WEIGHT["CRITICAL"],
    ),
    KeywordRule(
        name="bare_truncate",
        pattern=re.compile(r"\bTRUNCATE\s+TABLE\b", re.IGNORECASE),
        attack_type="piggy_back",
        severity="HIGH",
        description="TRUNCATE TABLE – data-destruction",
        weight=_SEVERITY_WEIGHT["HIGH"],
    ),

    # ── UNION-based injection ────────────────────────────────────────────
    # NOTE: bare UNION SELECT is legitimate SQL (used in Spider reporting queries).
    # We only flag when combined with injection markers:
    #   • NULL padding columns:  UNION SELECT NULL, NULL, ...
    #   • Followed by comment:   UNION SELECT ... --
    #   • Information schema:    UNION SELECT table_name FROM information_schema
    KeywordRule(
        name="union_select_null",
        pattern=re.compile(
            r"\bUNION\s+(ALL\s+)?SELECT\s+NULL\b", re.IGNORECASE
        ),
        attack_type="union_injection",
        severity="CRITICAL",
        description="UNION SELECT NULL – column-count probing injection",
        weight=_SEVERITY_WEIGHT["CRITICAL"],
    ),
    KeywordRule(
        name="union_select_info_schema",
        pattern=re.compile(
            r"\bUNION\s+(ALL\s+)?SELECT\b.{0,120}\binformation_schema\b",
            re.IGNORECASE | re.DOTALL,
        ),
        attack_type="union_injection",
        severity="CRITICAL",
        description="UNION SELECT ... information_schema – schema enumeration",
        weight=_SEVERITY_WEIGHT["CRITICAL"],
    ),
    KeywordRule(
        name="union_select_then_comment",
        pattern=re.compile(
            r"\bUNION\s+(ALL\s+)?SELECT\b.{0,200}--",
            re.IGNORECASE | re.DOTALL,
        ),
        attack_type="union_injection",
        severity="HIGH",
        description="UNION SELECT followed by -- comment – trailing comment injection",
        weight=_SEVERITY_WEIGHT["HIGH"],
    ),

    # ── Additional high-risk patterns ────────────────────────────────────
    KeywordRule(
        name="xp_cmdshell",
        pattern=re.compile(r"\bXP_CMDSHELL\b", re.IGNORECASE),
        attack_type="rce",
        severity="CRITICAL",
        description="xp_cmdshell – OS command execution via MSSQL",
        weight=_SEVERITY_WEIGHT["CRITICAL"],
    ),
    KeywordRule(
        name="load_file",
        pattern=re.compile(r"\bLOAD_FILE\s*\(", re.IGNORECASE),
        attack_type="file_access",
        severity="HIGH",
        description="LOAD_FILE() – reads server filesystem",
        weight=_SEVERITY_WEIGHT["HIGH"],
    ),
    KeywordRule(
        name="into_outfile",
        pattern=re.compile(r"\bINTO\s+OUTFILE\b", re.IGNORECASE),
        attack_type="file_access",
        severity="HIGH",
        description="INTO OUTFILE – writes query result to filesystem",
        weight=_SEVERITY_WEIGHT["HIGH"],
    ),
]


# ---------------------------------------------------------------------------
# Filter result dataclass
# ---------------------------------------------------------------------------

@dataclass
class KeywordFilterResult:
    """Result produced by :class:`KeywordFilter`."""

    flagged: bool
    """True when at least one rule matched."""

    keyword_risk_score: float
    """0.0–1.0 risk score derived from matched rule severities."""

    matched_rules: List[str] = field(default_factory=list)
    """Names of rules that fired."""

    attack_types_found: List[str] = field(default_factory=list)
    """Deduplicated ToxicSQL attack category names."""

    severities_found: List[str] = field(default_factory=list)
    """Deduplicated severity levels of matched rules."""

    match_details: List[Dict] = field(default_factory=list)
    """Per-match detail dicts: {rule, attack_type, severity, snippet}."""

    recommendation: str = "ALLOW"
    """Suggested action: ALLOW | FLAG_FOR_REVIEW | REJECT."""


# ---------------------------------------------------------------------------
# Main filter class
# ---------------------------------------------------------------------------

class KeywordFilter:
    """
    Paper-aligned keyword-based malicious content filter for ToxicSQL.

    The filter is the first (fastest) gate in the pipeline — it runs in
    O(n·r) where n = query length, r = number of rules (~22), all in pure
    Python / compiled regex.  Positive matches are passed to the AST-based
    detector for structural confirmation.

    Usage::

        kf = KeywordFilter()
        result = kf.filter(sql_query)
        if result.flagged:
            # escalate to AST / full ASTSafeSQL pipeline
            ...
    """

    def __init__(
        self,
        custom_rules: Optional[List[KeywordRule]] = None,
        strict_mode: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        custom_rules:
            Optional list of additional :class:`KeywordRule` objects to append
            to the built-in catalogue.
        strict_mode:
            When *True* (default), even MEDIUM-severity matches cause
            ``recommendation = FLAG_FOR_REVIEW``.  When *False*, only HIGH/
            CRITICAL matches are flagged.
        """
        self._rules = list(_RULES)
        if custom_rules:
            self._rules.extend(custom_rules)
        self.strict_mode = strict_mode
        logger.info(
            "KeywordFilter initialised with %d rules (strict_mode=%s)",
            len(self._rules),
            strict_mode,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def filter(self, sql_query: str) -> KeywordFilterResult:
        """
        Scan *sql_query* against all keyword rules.

        Parameters
        ----------
        sql_query:
            The SQL string generated by the Text-to-SQL model.

        Returns
        -------
        KeywordFilterResult
            Full match report including risk score and recommendation.
        """
        if not sql_query or not sql_query.strip():
            return KeywordFilterResult(
                flagged=False,
                keyword_risk_score=0.0,
                recommendation="ALLOW",
            )

        matched_rules: List[str] = []
        attack_types: List[str] = []
        severities: List[str] = []
        details: List[Dict] = []
        cumulative_weight: float = 0.0

        for rule in self._rules:
            m = rule.pattern.search(sql_query)
            if m is None:
                continue

            # Capture a short snippet around the match for reporting
            start = max(0, m.start() - 10)
            end = min(len(sql_query), m.end() + 20)
            snippet = sql_query[start:end].replace("\n", " ")

            matched_rules.append(rule.name)
            attack_types.append(rule.attack_type)
            severities.append(rule.severity)
            details.append(
                {
                    "rule": rule.name,
                    "attack_type": rule.attack_type,
                    "severity": rule.severity,
                    "description": rule.description,
                    "snippet": f"...{snippet}...",
                }
            )
            cumulative_weight += rule.weight

        flagged = len(matched_rules) > 0

        # Normalise risk score by the highest possible single-rule weight (CRITICAL=0.80),
        # NOT by total rule count. This ensures that even 1 CRITICAL match → score ≈ 1.0,
        # rather than 0.80/22 ≈ 0.036 which is too small to be useful as a signal.
        MAX_RULE_WEIGHT = _SEVERITY_WEIGHT["CRITICAL"]   # 0.80
        if flagged:
            # Use the max matched rule weight as the primary signal,
            # then cap cumulative at 1.0
            max_matched_weight = max(
                _SEVERITY_WEIGHT.get(sev, 0.15) for sev in severities
            )
            keyword_risk_score = min(1.0, max(max_matched_weight, cumulative_weight / MAX_RULE_WEIGHT))
        else:
            keyword_risk_score = 0.0

        recommendation = self._recommend(flagged, keyword_risk_score, severities)

        result = KeywordFilterResult(
            flagged=flagged,
            keyword_risk_score=keyword_risk_score,
            matched_rules=matched_rules,
            attack_types_found=list(dict.fromkeys(attack_types)),   # preserve order, deduplicate
            severities_found=list(dict.fromkeys(severities)),
            match_details=details,
            recommendation=recommendation,
        )

        if flagged:
            logger.debug(
                "KeywordFilter flagged | rules=%s | risk=%.2f | action=%s",
                matched_rules, keyword_risk_score, recommendation,
            )
        else:
            logger.debug("KeywordFilter: query passed (no matches)")

        return result

    def explain(self, sql_query: str) -> str:
        """
        Human-readable explanation of why *sql_query* was (or was not) flagged.
        """
        result = self.filter(sql_query)
        if not result.flagged:
            return "✅ Query passed all keyword rules — no suspicious patterns detected."

        lines = [
            f"🚨 Query FLAGGED — risk score: {result.keyword_risk_score:.2f}",
            f"   Recommendation: {result.recommendation}",
            f"   Attack types detected: {', '.join(result.attack_types_found)}",
            "",
            "   Matched rules:",
        ]
        for d in result.match_details:
            lines.append(
                f"     [{d['severity']:8s}] {d['rule']:40s} → {d['snippet']}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _recommend(
        self,
        flagged: bool,
        risk_score: float,
        severities: List[str],
    ) -> str:
        if not flagged:
            return "ALLOW"
        has_critical = "CRITICAL" in severities
        has_high = "HIGH" in severities
        if has_critical:
            return "REJECT"
        if has_high:
            return "REJECT" if self.strict_mode else "FLAG_FOR_REVIEW"
        return "FLAG_FOR_REVIEW"
