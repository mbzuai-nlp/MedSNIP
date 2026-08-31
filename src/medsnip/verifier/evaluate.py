"""Compute verifier metrics from saved verdicts — no LLM calls.

Walks data/8-verifier/**/verdicts_<split>_<source>.json and produces
data/8-verifier/metrics.json with per-(version × split × source × subset ×
ground_truth × abstain_mode) metrics.

Headline metric: F1 on the FALSE class against gold field `label_atomic`
with abstained verdicts dropped from the eval set.

For each cell:
  - n, n_true_gold, n_false_gold, n_abstain
  - accuracy
  - precision_F, recall_F, f1_F  (false class — main signal)
  - precision_T, recall_T, f1_T
  - macro_f1

Abstain modes:
  - drop      : exclude abstentions from numerator/denominator
  - as_false  : count an abstain as a 'false' prediction
  - as_true   : count an abstain as a 'true' prediction

Usage:
  python -m src.medsnip.verifier.evaluate
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
VERIFIER_DIR = ROOT / "data" / "8-verifier"
SPLIT_PATH = ROOT / "data" / "4-split" / "medsnip-bench.json"
BASELINE_PATH = (ROOT / "data" / "5-baselines" / "predictions" / "snippet" /
                 "full-context" / "gpt-5.4-high" / "predictions.json")
OUT_PATH = VERIFIER_DIR / "metrics.json"

GT_FIELDS = ["label_atomic", "label_human_general", "label_human_contextual"]
ABSTAIN_MODES = ["drop", "as_false", "as_true"]
# (verifier_confidence_threshold, baseline_fallback) — name the strategy via this tuple.
# Threshold None ⇒ no calibration. cap_hit + abstained + prediction=None always treated as "not addressed".
CALIBRATED_STRATEGIES = [
    (0.85, False),  # NA dropped at conf>=0.85
    (0.90, False),  # NA dropped at conf>=0.90
    (0.85, True),   # ensemble: v2 if conf>=0.85 else baseline
    (0.90, True),   # ensemble: v2 if conf>=0.90 else baseline  ← headline
]
SUBSETS = ("all", "consumer", "vignette")
SPLITS = ("all", "train", "dev", "test")

FNAME_RE = re.compile(r"^verdicts_(?P<split>train|dev|test)_(?P<source>[A-Za-z0-9_-]+)\.json$")


def _resolve_pred(rec: dict, abstain_mode: str) -> bool | None:
    """Map a verdict record to a binary prediction, or None to skip it."""
    if rec["abstained"]:
        if abstain_mode == "drop":
            return None
        if abstain_mode == "as_false":
            return False
        if abstain_mode == "as_true":
            return True
    if rec["prediction"] is None:
        # unresolved (parse failure, timeout, etc.) — always skip
        return None
    return bool(rec["prediction"])


def _resolve_calibrated(rec: dict, threshold: float,
                        baseline_pred: bool | None) -> bool | None:
    """Calibrated reading of a verdict.

    "Not addressed" if abstained / cap_hit / prediction None / confidence below
    threshold. If baseline_pred is provided, fall back to it on Not-addressed
    so the ensemble has full coverage; otherwise drop the record.
    """
    is_na = (rec["abstained"]
             or rec.get("cap_hit", False)
             or rec["prediction"] is None)
    if not is_na:
        c = rec.get("confidence")
        if c is None or c < threshold:
            is_na = True
    if is_na:
        return baseline_pred  # may itself be None ⇒ drop
    return bool(rec["prediction"])


def _f1(rows: list[dict], gt_field: str, abstain_mode: str) -> dict:
    tp_f = fp_f = fn_f = 0
    tp_t = fp_t = fn_t = 0
    n_used = n_correct = 0
    n_abstain = 0
    n_true_gold = n_false_gold = 0
    for r in rows:
        if r["abstained"]:
            n_abstain += 1
        pred = _resolve_pred(r, abstain_mode)
        if pred is None:
            continue
        gold = bool(r[gt_field])
        if gold:
            n_true_gold += 1
        else:
            n_false_gold += 1
        n_used += 1
        if pred == gold:
            n_correct += 1
        if gold is False and pred is False:    tp_f += 1
        elif gold is False and pred is True:   fn_f += 1
        elif gold is True and pred is False:   fp_f += 1
        if gold is True and pred is True:      tp_t += 1
        elif gold is True and pred is False:   fn_t += 1
        elif gold is False and pred is True:   fp_t += 1

    def prf(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        return round(p, 4), round(r, 4), round(f, 4)

    p_f, r_f, f1_f = prf(tp_f, fp_f, fn_f)
    p_t, r_t, f1_t = prf(tp_t, fp_t, fn_t)
    return {
        "n":             n_used,
        "n_abstain":     n_abstain,
        "n_true_gold":   n_true_gold,
        "n_false_gold":  n_false_gold,
        "accuracy":      round(n_correct / n_used, 4) if n_used else 0.0,
        "precision_F":   p_f, "recall_F": r_f, "f1_F": f1_f,
        "precision_T":   p_t, "recall_T": r_t, "f1_T": f1_t,
        "macro_f1":      round((f1_f + f1_t) / 2, 4),
    }


def _select(rows: list[dict], split: str, subset: str) -> list[dict]:
    out = []
    for r in rows:
        if split != "all" and r.get("split") != split:
            continue
        if subset != "all" and r.get("subset") != subset:
            continue
        out.append(r)
    return out


def _compute_table(joined: list[dict]) -> dict:
    """Per split × subset × gt × abstain_mode metrics."""
    table: dict = {}
    for sp in SPLITS:
        sp_key = "overall" if sp == "all" else sp
        table[sp_key] = {}
        for sub in SUBSETS:
            sub_key = sub if sub != "all" else "all"
            rows = _select(joined, sp, sub)
            table[sp_key][sub_key] = {
                gt: {mode: _f1(rows, gt, mode) for mode in ABSTAIN_MODES}
                for gt in GT_FIELDS
            }
    return table


def _f1_calibrated(rows: list[dict], gt_field: str,
                   threshold: float, baseline_fallback: bool) -> dict:
    """Same shape as _f1, but uses calibrated thresholding with optional
    baseline fallback (records carry a `_baseline_pred` field — populated by
    main() when BASELINE_PATH exists)."""
    tp_f = fp_f = fn_f = 0
    tp_t = fp_t = fn_t = 0
    n_used = n_correct = 0
    n_na = 0  # not-addressed before fallback
    n_true_gold = n_false_gold = 0
    for r in rows:
        bp = r.get("_baseline_pred") if baseline_fallback else None
        pred = _resolve_calibrated(r, threshold, bp)
        # Track NA before fallback for diagnostics
        is_na = (r["abstained"] or r.get("cap_hit", False)
                 or r["prediction"] is None
                 or (r.get("confidence") is None or
                     (r.get("confidence") or 0) < threshold))
        if is_na:
            n_na += 1
        if pred is None:
            continue
        gold = bool(r[gt_field])
        if gold:
            n_true_gold += 1
        else:
            n_false_gold += 1
        n_used += 1
        if pred == gold:
            n_correct += 1
        if gold is False and pred is False:    tp_f += 1
        elif gold is False and pred is True:   fn_f += 1
        elif gold is True and pred is False:   fp_f += 1
        if gold is True and pred is True:      tp_t += 1
        elif gold is True and pred is False:   fn_t += 1
        elif gold is False and pred is True:   fp_t += 1

    def prf(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        return round(p, 4), round(r, 4), round(f, 4)

    p_f, r_f, f1_f = prf(tp_f, fp_f, fn_f)
    p_t, r_t, f1_t = prf(tp_t, fp_t, fn_t)
    return {
        "n":             n_used,
        "n_na_pre_fb":   n_na,
        "n_true_gold":   n_true_gold,
        "n_false_gold":  n_false_gold,
        "accuracy":      round(n_correct / n_used, 4) if n_used else 0.0,
        "precision_F":   p_f, "recall_F": r_f, "f1_F": f1_f,
        "precision_T":   p_t, "recall_T": r_t, "f1_T": f1_t,
        "macro_f1":      round((f1_f + f1_t) / 2, 4),
    }


def _compute_calibrated_table(joined: list[dict]) -> dict:
    """Per (threshold, baseline_fallback) × split × subset × gt metrics."""
    table: dict = {}
    for thr, fb in CALIBRATED_STRATEGIES:
        key = f"conf>={thr}{' +baseline-fallback' if fb else ' NA-dropped'}"
        table[key] = {}
        for sp in SPLITS:
            sp_key = "overall" if sp == "all" else sp
            table[key][sp_key] = {}
            for sub in SUBSETS:
                sub_key = sub if sub != "all" else "all"
                rows = _select(joined, sp, sub)
                table[key][sp_key][sub_key] = {
                    gt: _f1_calibrated(rows, gt, thr, fb) for gt in GT_FIELDS
                }
    return table


def _iter_verdict_files() -> list[tuple[str, str, str, Path]]:
    """Yield (version, split, source, path). version='' for files at the root."""
    out = []
    for path in sorted(VERIFIER_DIR.rglob("verdicts_*.json")):
        m = FNAME_RE.match(path.name)
        if not m:
            continue
        rel_parent = path.parent.relative_to(VERIFIER_DIR)
        version = "" if rel_parent == Path(".") else str(rel_parent)
        out.append((version, m["split"], m["source"], path))
    return out


def main():
    if not SPLIT_PATH.exists():
        raise SystemExit(f"missing {SPLIT_PATH}")
    gold_rows = json.loads(SPLIT_PATH.read_text())
    gold: dict[str, dict] = {r["snippet_id"]: r for r in gold_rows}

    # Optional baseline preds (for calibrated-ensemble fallback)
    baseline_pred: dict[str, bool] = {}
    if BASELINE_PATH.exists():
        for b in json.loads(BASELINE_PATH.read_text()):
            sid = b.get("snippet_id") or b.get("id")
            if sid is not None and "prediction" in b:
                baseline_pred[sid] = bool(b["prediction"])

    results: dict = {}
    headline_rows: list[tuple] = []
    calibrated_rows: list[tuple] = []

    for version, split, source, path in _iter_verdict_files():
        records = json.loads(path.read_text())
        joined = []
        for rec in records:
            g = gold.get(rec["id"])
            if g is None:
                continue
            joined.append({
                **rec,
                "subset": g["subset"],
                "split":  g["split"],
                "label_atomic":           g["label_atomic"],
                "label_human_general":    g["label_human_general"],
                "label_human_contextual": g["label_human_contextual"],
                "_baseline_pred":         baseline_pred.get(rec["id"]),
            })
        ver_key = version or "_root"
        run_key = f"{split}_{source}"
        results.setdefault(ver_key, {})[run_key] = {
            "n_records":   len(records),
            "n_joined":    len(joined),
            "n_unresolved": sum(1 for r in records
                                if not r["abstained"] and r["prediction"] is None),
            "table":       _compute_table(joined),
            "calibrated":  _compute_calibrated_table(joined),
        }
        cell = results[ver_key][run_key]["table"]["dev"]["all"]["label_atomic"]["drop"]
        headline_rows.append((ver_key, run_key, cell))
        for thr, fb in CALIBRATED_STRATEGIES:
            key = f"conf>={thr}{' +baseline-fallback' if fb else ' NA-dropped'}"
            ccell = results[ver_key][run_key]["calibrated"][key]["dev"]["all"]["label_atomic"]
            calibrated_rows.append((ver_key, run_key, key, ccell))

    VERIFIER_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))

    print("\nHeadline (dev, all subsets, gt=label_atomic, abstain=drop):")
    print(f"{'version':<10s} {'run':<22s} {'n':>5s} {'acc':>6s} "
          f"{'P_F':>6s} {'R_F':>6s} {'F1_F':>6s} {'F1_T':>6s} {'macro':>6s}")
    for ver_key, run_key, cell in headline_rows:
        if cell["n"] == 0:
            continue
        print(f"{ver_key:<10s} {run_key:<22s} {cell['n']:>5d} {cell['accuracy']:>6.3f} "
              f"{cell['precision_F']:>6.3f} {cell['recall_F']:>6.3f} "
              f"{cell['f1_F']:>6.3f} {cell['f1_T']:>6.3f} {cell['macro_f1']:>6.3f}")

    print("\nCalibrated strategies (dev, all subsets, gt=label_atomic):")
    print(f"{'version':<10s} {'run':<22s} {'strategy':<30s} {'n':>5s} "
          f"{'acc':>6s} {'F1_F':>6s} {'F1_T':>6s} {'macro':>6s}")
    for ver_key, run_key, strategy, cell in calibrated_rows:
        if cell["n"] == 0:
            continue
        print(f"{ver_key:<10s} {run_key:<22s} {strategy:<30s} {cell['n']:>5d} "
              f"{cell['accuracy']:>6.3f} {cell['f1_F']:>6.3f} {cell['f1_T']:>6.3f} "
              f"{cell['macro_f1']:>6.3f}")

    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
