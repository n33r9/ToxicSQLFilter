"""
eval_toxicsql.py — Evaluation of ToxicSQLFilter  (Train / Validate / Test)
===========================================================================
Runs ToxicSQLFilter (Keyword + AST) against the labelled dataset
train_spider_tau_com_time.json (field "toxic": 0|1).

Protocol  (3-way stratified split, zero data leakage)
------------------------------------------------------
  Split   :  60 % TRAIN  |  20 % VALIDATE  |  20 % TEST
             (stratified — preserves toxic/clean ratio in every split)

  Step 1  :  Run filter on TRAIN   — observe score distribution (informational)
  Step 2  :  Sweep threshold 0.00–1.00 on VALIDATE → pick θ* that maximises F1
             *** threshold is selected on VALIDATE, NEVER on TEST ***
  Step 3  :  Apply θ* once to TEST → report final generalisation metrics

Output
------
  evaluation_results.json  — per-query scores + all metric tables
  evaluation_results.csv   — tabular rows (split column: train/val/test)
"""

import json
import csv
import os
import logging
import random
from time import time
from typing import Dict, List, Tuple

from toxic_sql_filter import ToxicSQLFilter

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
_HERE     = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_HERE, "train_spider_tau_com_time.json")
OUT_JSON  = os.path.join(_HERE, "evaluation_results.json")
OUT_CSV   = os.path.join(_HERE, "evaluation_results.csv")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("eval_toxicsql")

RANDOM_SEED  = 42
TRAIN_RATIO  = 0.60   # 60 %
VAL_RATIO    = 0.20   # 20 %  → test gets the remaining 20 %


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Stratified 3-way split
# ---------------------------------------------------------------------------

def stratified_split_3way(
    data: List[Dict],
    train_ratio: float = 0.60,
    val_ratio:   float = 0.20,
    seed:        int   = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Stratified split into TRAIN / VALIDATE / TEST.
    Preserves the toxic:clean ratio independently in each split.

    Returns
    -------
    (train_items, val_items, test_items)
    """
    assert train_ratio + val_ratio < 1.0, "train + val must be < 1.0"
    rng = random.Random(seed)

    toxic_items = [x for x in data if int(x.get("toxic", 0)) == 1]
    clean_items = [x for x in data if int(x.get("toxic", 0)) == 0]

    rng.shuffle(toxic_items)
    rng.shuffle(clean_items)

    def split3(lst: List) -> Tuple[List, List, List]:
        n = len(lst)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)
        return lst[:n_train], lst[n_train:n_train + n_val], lst[n_train + n_val:]

    toxic_tr, toxic_va, toxic_te = split3(toxic_items)
    clean_tr, clean_va, clean_te = split3(clean_items)

    train = toxic_tr + clean_tr;  rng.shuffle(train)
    val   = toxic_va + clean_va;  rng.shuffle(val)
    test  = toxic_te + clean_te;  rng.shuffle(test)

    return train, val, test


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _conf(y_true: List[int], preds: List[int]) -> Tuple[int, int, int, int]:
    tp = fp = tn = fn = 0
    for y, p in zip(y_true, preds):
        if   y == 1 and p == 1: tp += 1
        elif y == 0 and p == 1: fp += 1
        elif y == 0 and p == 0: tn += 1
        else:                   fn += 1
    return tp, fp, tn, fn


def compute_metrics(y_true: List[int], preds: List[int], label: str = "") -> Dict:
    """Binary classification metrics from hard 0/1 predictions."""
    tp, fp, tn, fn = _conf(y_true, preds)
    prec   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1  = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn)     if (tp + tn + fp + fn) > 0 else 0.0
    return dict(label=label, precision=prec, recall=recall,
                f1=f1, accuracy=acc, tp=tp, fp=fp, tn=tn, fn=fn)


def threshold_sweep(y_true: List[int], scores: List[float]) -> Dict:
    """
    Sweep θ ∈ [0.00, 1.00] (step 0.01).
    Prediction = 1 (toxic) when score > θ.
    Returns the operating point with the highest F1.
    """
    best: Dict = {"threshold": None, "f1": -1.0}
    for t_int in range(0, 101):
        t = t_int / 100.0
        preds = [1 if s > t else 0 for s in scores]
        tp, fp, tn, fn = _conf(y_true, preds)
        prec   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0.0
        if f1 > best["f1"]:
            best = dict(threshold=t, f1=f1, precision=prec, recall=recall,
                        tp=tp, fp=fp, tn=tn, fn=fn)
    return best


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def print_header(title: str, n: int) -> None:
    bar = "═" * 70
    print(f"\n{bar}")
    print(f"  {title}  (n={n})")
    print(bar)


def print_metrics(label: str, m: Dict, threshold=None) -> None:
    bar = "─" * 70
    print(f"\n{bar}")
    print(f"  {label}")
    print(bar)
    if threshold is not None:
        print(f"  Threshold : {threshold:.2f}")
    print(f"  Precision : {m['precision']:.4f}")
    print(f"  Recall    : {m['recall']:.4f}")
    print(f"  F1        : {m['f1']:.4f}")
    print(f"  Accuracy  : {m['accuracy']:.4f}")
    print(f"  TP={m['tp']:4d}  FP={m['fp']:4d}  TN={m['tn']:4d}  FN={m['fn']:4d}")
    print(bar)


# ---------------------------------------------------------------------------
# Filter runner
# ---------------------------------------------------------------------------

def run_filter(
    filt: ToxicSQLFilter,
    items: List[Dict],
    split_name: str,
) -> List[Dict]:
    """Run ToxicSQLFilter over *items*; return annotated row dicts."""
    rows: List[Dict] = []
    bar = tqdm(total=len(items), desc=f"  Filtering [{split_name:5s}]") if tqdm else None

    for i, item in enumerate(items):
        sql   = item.get("query",    "")
        nlq   = item.get("question", "")
        label = int(item.get("toxic", 0))

        try:
            r         = filt.filter(sql=sql, nlq=nlq)
            score     = float(r.overall_risk_score)
            is_toxic  = bool(r.is_toxic)
            action    = r.action
            kw_rules  = ",".join(r.keyword_matched_rules)
            kw_types  = ",".join(r.keyword_attack_types)
            ast_patts = ",".join(r.ast_suspicious_patterns)
        except Exception as exc:
            logger.exception("Error item %d [%s]: %s", i, split_name, exc)
            score, is_toxic, action = 1.0, True, "REJECT"
            kw_rules = kw_types = ast_patts = ""

        rows.append({
            "split":    split_name,
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
            # filled in later
            "pred_sweep": 0,
            "pred_hard":  1 if is_toxic else 0,
        })

        if bar: bar.update(1)
        if (i + 1) % 1000 == 0:
            logger.info("  [%s] %d / %d", split_name, i + 1, len(items))

    if bar: bar.close()
    return rows


def apply_threshold(rows: List[Dict], threshold: float) -> None:
    """In-place: set pred_sweep for each row."""
    for r in rows:
        r["pred_sweep"] = 1 if r["score"] > threshold else 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_evaluation() -> None:
    t0 = time()

    # ── 1. Load ────────────────────────────────────────────────────────────
    logger.info("Loading dataset: %s", DATA_PATH)
    data    = load_dataset(DATA_PATH)
    n_total = len(data)
    n_toxic = sum(1 for x in data if int(x.get("toxic", 0)) == 1)
    n_clean = n_total - n_toxic
    logger.info(
        "Dataset: %d total  (%d toxic [%.1f%%], %d clean)",
        n_total, n_toxic, 100 * n_toxic / n_total, n_clean,
    )

    # ── 2. Stratified 3-way split ──────────────────────────────────────────
    train_items, val_items, test_items = stratified_split_3way(
        data,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        seed=RANDOM_SEED,
    )

    def _stats(items):
        nt = sum(1 for x in items if int(x.get("toxic", 0)) == 1)
        nc = len(items) - nt
        return f"n={len(items)}, toxic={nt}, clean={nc}"

    logger.info("TRAIN    : %s", _stats(train_items))
    logger.info("VALIDATE : %s", _stats(val_items))
    logger.info("TEST     : %s", _stats(test_items))

    # ── 3. Initialise filter ───────────────────────────────────────────────
    logger.info("Initialising ToxicSQLFilter …")
    filt = ToxicSQLFilter(
        ast_alignment_threshold=0.6,
        ast_suspicious_weight=0.35,
        flag_threshold=0.20,
        reject_threshold=0.40,
    )

    # ── 4. Run filter on all three splits ──────────────────────────────────
    train_rows = run_filter(filt, train_items, "train")
    val_rows   = run_filter(filt, val_items,   "val")
    test_rows  = run_filter(filt, test_items,  "test")

    # ── 5. Threshold selection on VALIDATE only ────────────────────────────
    logger.info("Sweeping thresholds on VALIDATE set …")
    y_val    = [r["label"] for r in val_rows]
    s_val    = [r["score"] for r in val_rows]
    sweep_val = threshold_sweep(y_val, s_val)
    best_threshold = sweep_val["threshold"]

    logger.info(
        "Best threshold θ* = %.2f  (VAL F1=%.4f, P=%.4f, R=%.4f)",
        best_threshold, sweep_val["f1"], sweep_val["precision"], sweep_val["recall"],
    )

    # ── 6. Apply θ* to all splits ─────────────────────────────────────────
    apply_threshold(train_rows, best_threshold)
    apply_threshold(val_rows,   best_threshold)
    apply_threshold(test_rows,  best_threshold)

    # ── 7. Compute metrics ─────────────────────────────────────────────────

    def rows_metrics(rows, label_prefix):
        y     = [r["label"]      for r in rows]
        p_sw  = [r["pred_sweep"] for r in rows]
        p_hd  = [r["pred_hard"]  for r in rows]
        return (
            compute_metrics(y, p_sw, label=f"{label_prefix} sweep@θ*"),
            compute_metrics(y, p_hd, label=f"{label_prefix} hard (action≠ALLOW)"),
        )

    m_train_sw, m_train_hd = rows_metrics(train_rows, "TRAIN")
    m_val_sw,   m_val_hd   = rows_metrics(val_rows,   "VAL  ")
    m_test_sw,  m_test_hd  = rows_metrics(test_rows,  "TEST ")

    # Oracle: best possible F1 if threshold were selected ON test (shows upper bound)
    y_test   = [r["label"] for r in test_rows]
    s_test   = [r["score"] for r in test_rows]
    sweep_test_oracle = threshold_sweep(y_test, s_test)

    # ── 8. Print results ───────────────────────────────────────────────────

    print_header("TRAIN SET", len(train_rows))
    print_metrics("sweep @ θ* (from VAL)",       m_train_sw, threshold=best_threshold)
    print_metrics("hard decision (action≠ALLOW)", m_train_hd)

    print_header("VALIDATE SET  ← threshold selected here", len(val_rows))
    print_metrics("sweep @ θ*  (best F1 on VAL)", m_val_sw, threshold=best_threshold)
    print_metrics("hard decision (action≠ALLOW)",  m_val_hd)

    print_header("TEST SET  ← FINAL generalisation result", len(test_rows))
    print_metrics(
        f"sweep @ θ*={best_threshold:.2f} (fixed from VAL)  ← PRIMARY METRIC",
        m_test_sw, threshold=best_threshold,
    )
    print_metrics("hard decision (action≠ALLOW)", m_test_hd)

    print(f"\n  [Oracle / upper bound]")
    print(f"  Best θ if selected ON TEST directly : "
          f"{sweep_test_oracle['threshold']:.2f}  "
          f"(F1={sweep_test_oracle['f1']:.4f})  "
          f"← data leakage — for reference only")

    # ── 9. Gap / overfitting analysis ─────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  OVERFITTING CHECK  (VAL → TEST gap, sweep@θ*)")
    print(f"{'─'*70}")
    for metric in ("f1", "precision", "recall", "accuracy"):
        gap = m_val_sw[metric] - m_test_sw[metric]
        flag = "  ⚠ possible overfit" if abs(gap) > 0.05 else "  ✓ OK"
        print(f"  Δ{metric:9s} = {gap:+.4f}{flag}")
    print(f"{'─'*70}")

    # ── 10. Save JSON ──────────────────────────────────────────────────────
    def _slim(r):
        return {k: r[k] for k in (
            "split", "db_id", "label", "score", "action",
            "kw_rules", "kw_types", "ast_patt", "pred_sweep", "pred_hard",
        )}

    out = {
        "config": {
            "random_seed":  RANDOM_SEED,
            "train_ratio":  TRAIN_RATIO,
            "val_ratio":    VAL_RATIO,
            "test_ratio":   round(1.0 - TRAIN_RATIO - VAL_RATIO, 2),
            "best_threshold_from_val": best_threshold,
        },
        "train": {
            "n":     len(train_rows),
            "sweep": m_train_sw,
            "hard":  m_train_hd,
        },
        "validate": {
            "n":          len(val_rows),
            "sweep":      sweep_val,          # the full sweep result
            "hard":       m_val_hd,
        },
        "test": {
            "n":                        len(test_rows),
            "sweep_with_val_threshold": m_test_sw,   # PRIMARY result
            "hard":                     m_test_hd,
            "sweep_oracle":             sweep_test_oracle,  # reference only
        },
        "rows": [_slim(r) for r in train_rows + val_rows + test_rows],
    }

    logger.info("Saving JSON → %s", OUT_JSON)
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    # ── 11. Save CSV ───────────────────────────────────────────────────────
    logger.info("Saving CSV  → %s", OUT_CSV)
    fields = [
        "split", "db_id", "label", "score", "action",
        "pred_sweep", "pred_hard",
        "kw_rules", "kw_types", "ast_patt", "sql", "nlq",
    ]
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in train_rows + val_rows + test_rows:
            w.writerow({
                "split":      r["split"],
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
    print(f"  Time : {elapsed:.1f}s\n")


if __name__ == "__main__":
    run_evaluation()
