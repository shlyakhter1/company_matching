#!/usr/bin/env python3
"""
benchmark.py — score every available matcher on the labeled pairs and report metrics.

Outputs (to --outdir, default ./out):
  pair_scores.csv  : every pair with each matcher's similarity score (for visualize.py)
  metrics.csv      : per-matcher best-F1 threshold, P/R/F1, ROC-AUC, avg-precision,
                     and a breakdown of where errors fall (hard vs easy negatives)
  records_scored.* : (written by visualize.py) — not here

Usage:
  python benchmark.py
  python benchmark.py --matchers fuzzy_token_sort_norm tfidf_char
  python benchmark.py --records data/company_records.csv --pairs data/company_pairs.csv
"""
import csv, os, argparse, json
import numpy as np
import matchers as M


def load_pairs(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    pairs = [(r["name_a"], r["name_b"]) for r in rows]
    y = np.array([int(r["label"]) for r in rows])
    ptype = [r["pair_type"] for r in rows]
    return rows, pairs, y, ptype


def prf(y, pred):
    """P/R/F1 (+ tp/fp/fn) at a fixed operating point — the one shared
    implementation for benchmark.py, llm_matcher.py, and hybrid_eval.py.
    Convention: no predicted positives -> P=1.0."""
    pred = np.asarray(pred, dtype=bool)
    y = np.asarray(y)
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    P = tp / (tp + fp) if tp + fp else 1.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    return P, R, F, tp, fp, fn


def best_threshold(y, scores):
    """Sweep candidate thresholds, return the one maximizing F1 plus the curve."""
    order = np.argsort(-scores)
    ths = np.unique(np.concatenate([[1.01], scores[order], [-0.01]]))
    best = (0.5, 0.0, 0.0, 0.0)  # thr, P, R, F1
    curve = []
    for t in ths:
        P, R, F, *_ = prf(y, scores >= t)
        curve.append((float(t), P, R, F))
        if F > best[3]:
            best = (float(t), P, R, F)
    return best, curve


def roc_auc(y, s):
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        return float(roc_auc_score(y, s)), float(average_precision_score(y, s))
    except Exception:
        return float("nan"), float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="data/company_records.csv")
    ap.add_argument("--pairs", default="data/company_pairs.csv")
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--matchers", nargs="*", default=None,
                    help="subset by .name; default = all available")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    rows, pairs, y, ptype = load_pairs(args.pairs)
    ptype = np.array(ptype)
    print(f"Loaded {len(pairs)} pairs ({int((y==1).sum())} pos / {int((y==0).sum())} neg)\n")

    registry = M.default_registry()
    if args.matchers:
        registry = [m for m in registry if m.name in args.matchers]

    score_cols = {}        # name -> np.ndarray
    metrics = []
    for m in registry:
        ok, why = M.available(m)
        if not ok:
            print(f"  SKIP  {m.name:22} ({why})")
            continue
        try:
            print(f"  run   {m.name:22} {why}")
            s = np.asarray(m.score_pairs(pairs), dtype=float)
        except Exception as e:
            print(f"  FAIL  {m.name:22} {e.__class__.__name__}: {e}")
            continue
        score_cols[m.name] = s
        (thr, P, R, F), _curve = best_threshold(y, s)
        auc, ap_ = roc_auc(y, s)
        pred = s >= thr
        fp_hard = int(np.sum(pred & (y == 0) & (ptype == "hard_negative")))
        fp_easy = int(np.sum(pred & (y == 0) & (ptype == "easy_negative")))
        fn = int(np.sum(~pred & (y == 1)))
        metrics.append(dict(matcher=m.name, threshold=round(thr, 4),
                            precision=round(P, 4), recall=round(R, 4), f1=round(F, 4),
                            roc_auc=round(auc, 4), avg_precision=round(ap_, 4),
                            false_merge_hard=fp_hard, false_merge_easy=fp_easy, missed_pos=fn))

    # write per-pair scores
    sp = os.path.join(args.outdir, "pair_scores.csv")
    with open(sp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        cols = list(score_cols.keys())
        w.writerow(["pair_id", "name_a", "name_b", "label", "pair_type",
                    "cluster_a", "cluster_b"] + cols)
        for i, r in enumerate(rows):
            w.writerow([r["pair_id"], r["name_a"], r["name_b"], r["label"], r["pair_type"],
                        r["cluster_a"], r["cluster_b"]] + [f"{score_cols[c][i]:.4f}" for c in cols])

    # write metrics
    mp = os.path.join(args.outdir, "metrics.csv")
    if metrics:
        keys = list(metrics[0].keys())
        with open(mp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(metrics)
        json.dump(metrics, open(os.path.join(args.outdir, "metrics.json"), "w"), indent=2)

    # console summary
    print("\n=== best-F1 operating point per matcher ===")
    print(f"{'matcher':22} {'thr':>5} {'P':>6} {'R':>6} {'F1':>6} {'AUC':>6}  {'FP_hard':>7} {'FP_easy':>7} {'miss':>5}")
    for r in sorted(metrics, key=lambda d: -d["f1"]):
        print(f"{r['matcher']:22} {r['threshold']:>5.2f} {r['precision']:>6.3f} {r['recall']:>6.3f} "
              f"{r['f1']:>6.3f} {r['roc_auc']:>6.3f}  {r['false_merge_hard']:>7} {r['false_merge_easy']:>7} {r['missed_pos']:>5}")
    print(f"\nwrote {sp}\nwrote {mp}")


if __name__ == "__main__":
    main()
