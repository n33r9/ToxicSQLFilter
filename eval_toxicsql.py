"""
eval_toxicsql.py — Evaluation of ToxicSQLFilter
================================================
Runs ToxicSQLFilter (Keyword + AST) against the labelled dataset
train_spider_tau_com_time.json (field "toxic": 0|1).

Steps:
  1. Run every query through ToxicSQLFilter
  2. Sweep thresholds 0.00–1.00 to find the operating point that maximises F1
  3. Report hard-decision metrics (action != ALLOW → predicted toxic)
  4. Save results to JSON + CSV

Output:
  evaluation_results.json   — per-query scores + best threshold sweep
  evaluation_results.csv    — tabular results
"""

import json
import csv
import os
import logging
from time import time
from typing import Dict, List

from toxic_sql_filter import ToxicSQLFilter

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE     = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_HERE, "train_spider_tau_com_time.json")
OUT_JSON  = os.path.join(_HERE, "evaluation_results.json")
OUT_CSV   = os.path.join(_HERE, "evaluation_results.csv")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("eval_toxicsql")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_dataset(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def threshold_sweep(y_true: List[int], scores: List[float]) -> Dict:
    """Sweep risk-score thresholds, return best F1 operating point.
    pred=1 (toxic) when score > threshold."""
    best: Dict = {"threshold": None, "f1": -1.0}
    for t_int in range(0, 101):
        t = t_int / 100.0
        tp = fp = tn = fn = 0
        for y, s in zip(y_true, scores):
            pred = 1 if s > t else 0
            if   y == 1 and pred == 1: tp += 1
            elif y == 0 and pred == 1: fp += 1
            elif y == 0 and pred == 0: tn += 1
            else:                      fn += 1
        prec   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0.0
        if f1 > best["f1"]:
            best = dict(threshold=t, f1=f1, precision=prec, recall=recall,
                        tp=tp, fp=fp, tn=tn, fn=fn)
    return best


def hard_metrics(y_true: List[int], preds: List[int]) -> Dict:
    """Metrics from binary predictions (action != ALLOW → 1)."""
    tp = fp = tn = fn = 0
    for y, p in zip(y_true, preds):
        if   y == 1 and p == 1: tp += 1
        elif y == 0 and p == 1: fp += 1
        elif y == 0 and p == 0: tn += 1
        else:                   fn += 1
    prec   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0.0
    return dict(threshold="hard", f1=f1, precision=prec, recall=recall,
                tp=tp, fp=fp, tn=tn, fn=fn)


def print_table(label: str, m: Dict) -> None:
    bar = "─" * 62
    t = m["threshold"]
    print(f"\n{bar}")
    print(f"  {label}")
    print(bar)
    print(f"  Threshold : {f'{t:.2f}' if isinstance(t, float) else t}")
    print(f"  F1        : {m['f1']:.4f}")
    print(f"  Precision : {m['precision']:.4f}")
    print(f"  Recall    : {m['recall']:.4f}")
    print(f"  TP={m['tp']}  FP={m['fp']}  TN={m['tn']}  FN={m['fn']}")
    print(bar)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_evaluation():
    t0 = time()

    logger.info("Loading dataset: %s", DATA_PATH)
    data = load_dataset(DATA_PATH)
    n = len(data)
    logger.info("Dataset: %d items  (%d toxic, %d clean)",
                n,
                sum(1 for x in data if int(x.get("toxic", 0)) == 1),
                sum(1 for x in data if int(x.get("toxic", 0)) == 0))

    logger.info("Initialising ToxicSQLFilter …")
    filt = ToxicSQLFilter(
        ast_alignment_threshold=0.6,
        ast_suspicious_weight=0.35,
        flag_threshold=0.20,
        reject_threshold=0.40,
    )

    rows: List[Dict] = []
    bar = tqdm(total=n, desc="Filtering") if tqdm else None

    for i, item in enumerate(data):
        sql   = item.get("query",    "")
        nlq   = item.get("question", "")
        label = int(item.get("toxic", 0))

        try:
            r = filt.filter(sql=sql, nlq=nlq)
            score      = float(r.overall_risk_score)
            is_toxic   = bool(r.is_toxic)
            action     = r.action
            kw_rules   = ",".join(r.keyword_matched_rules)
            kw_types   = ",".join(r.keyword_attack_types)
            ast_patts  = ",".join(r.ast_suspicious_patterns)
        except Exception as exc:
            logger.exception("Error item %d: %s", i, exc)
            score, is_toxic, action = 1.0, True, "REJECT"
            kw_rules = kw_types = ast_patts = ""

        rows.append({
            "index":    i,
            "db_id":    item.get("db_id", ""),
            "label":    label,
            "score":    score,
            "is_toxic": is_toxic,
            "action":   action,
            "kw_rules": kw_rules,
            "kw_types": kw_types,
            "ast_patt": ast_patts,
            "sql":      sql,
            "nlq":      nlq,
        })

        if bar: bar.update(1)
        if (i + 1) % 1000 == 0:
            logger.info("  %d / %d", i + 1, n)

    if bar: bar.close()

    # ── Metrics ────────────────────────────────────────────────────────────
    y_true = [r["label"]    for r in rows]
    scores = [r["score"]    for r in rows]
    preds  = [1 if r["is_toxic"] else 0 for r in rows]

    best_sweep = threshold_sweep(y_true, scores)
    best_hard  = hard_metrics(y_true, preds)

    # annotate sweep pred
    for r in rows:
        r["pred_sweep"] = 1 if r["score"] > best_sweep["threshold"] else 0
        r["pred_hard"]  = 1 if r["is_toxic"] else 0

    # ── Print ──────────────────────────────────────────────────────────────
    print_table("ToxicSQLFilter — threshold sweep (best F1)", best_sweep)
    print_table("ToxicSQLFilter — hard decision  (action ≠ ALLOW)", best_hard)

    # ── Save JSON ──────────────────────────────────────────────────────────
    logger.info("Saving JSON → %s", OUT_JSON)
    out_rows = [
        {k: r[k] for k in ("index","db_id","label","score","action",
                            "kw_rules","kw_types","ast_patt","pred_sweep","pred_hard")}
        for r in rows
    ]
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump({"sweep": best_sweep, "hard": best_hard, "rows": out_rows}, fh, indent=2)

    # ── Save CSV ───────────────────────────────────────────────────────────
    logger.info("Saving CSV  → %s", OUT_CSV)
    fields = ["index","db_id","label","score","action",
              "pred_sweep","pred_hard","kw_rules","kw_types","ast_patt","sql","nlq"]
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({
                "index":      r["index"],
                "db_id":      r["db_id"],
                "label":      r["label"],
                "score":      f"{r['score']:.4f}",
                "action":     r["action"],
                "pred_sweep": r["pred_sweep"],
                "pred_hard":  r["pred_hard"],
                "kw_rules":   r["kw_rules"][:200],
                "kw_types":   r["kw_types"][:100],
                "ast_patt":   r["ast_patt"][:200],
                "sql":        (r["sql"] or "")[:500],
                "nlq":        (r["nlq"] or "")[:200],
            })

    elapsed = time() - t0
    logger.info("Done in %.1fs", elapsed)
    print(f"\n  JSON → {OUT_JSON}")
    print(f"  CSV  → {OUT_CSV}")
    print(f"  Time : {elapsed:.1f}s")


if __name__ == "__main__":
    run_evaluation()
