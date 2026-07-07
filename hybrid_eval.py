#!/usr/bin/env python3
"""
hybrid_eval.py — compose a router score + stored LLM band decisions into the
hybrid pipeline score, and record it alongside the plain matchers (Round 4).

Composite score per pair (band = [--band-lo, --band-hi], default 0.10/0.95):
  in band, LLM said same      -> 0.98
  in band, LLM said different -> 0.02
  in band, no stored decision -> router score (counted and reported)
  outside band                -> router score

Outputs (all idempotent — re-running replaces the same column/row):
  <outdir>/pair_scores.csv    gains a hybrid_<name>_llm column
  <outdir>/metrics.csv        gains a hybrid_<name>_llm row; fixed-band P/R/F1
                              at threshold 0.5, ROC-AUC/AP from the composite
  <outdir>/hybrid_stats.json  band stats per hybrid (size, reuse, accuracy, sweep)
  <outdir>/router_comparison.md   comparison table across all recorded hybrids

benchmark.py rewrites pair_scores.csv/metrics.csv wholesale, so re-run this
script after any benchmark.py run to restore the hybrid columns.

Usage:
  python hybrid_eval.py --router tfidf --decisions out/llm_band_decisions.csv --name tfidf
  python hybrid_eval.py --router alias --decisions out/llm_band2_decisions.csv --name alias
  python hybrid_eval.py --router lt_comp_en --decisions out/llm_band3_decisions.csv --name ltcomp
"""
import argparse, csv, json, os
import numpy as np

from benchmark import best_threshold, load_pairs, prf, roc_auc
from llm_matcher import build_blocker, load_reuse


def composite_scores(rows, router_scores, decisions, lo, hi):
    comp = np.array(router_scores, dtype=float)
    in_band = (comp >= lo) & (comp <= hi)
    missing = 0
    for i, r in enumerate(rows):
        if not in_band[i]:
            continue
        d = decisions.get(r["pair_id"])
        if d is None:
            missing += 1
        else:
            comp[i] = 0.98 if d else 0.02
    return comp, in_band, missing


def upsert_column(path, pair_ids, colname, values):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    if len(rows) != len(pair_ids) or any(r["pair_id"] != p for r, p in zip(rows, pair_ids)):
        raise SystemExit(f"{path} does not match the pairs file — run benchmark.py first")
    for r, v in zip(rows, values):
        r[colname] = f"{v:.4f}"
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def upsert_row(path, row):
    rows = [r for r in csv.DictReader(open(path, encoding="utf-8"))
            if r["matcher"] != row["matcher"]]
    rows.append({k: str(v) for k, v in row.items()})
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def render_comparison(stats_path, md_path):
    stats = json.load(open(stats_path))
    lines = ["# Hybrid router comparison (router + LLM band, fixed band per stats)", "",
             "| pipeline | router | band | reused | LLM band acc | P | R | F1 | misses | FP | best-F1 sweep | ROC-AUC |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, s in stats.items():
        lines.append(
            f"| {name} | {s['router']} | {s['band_size']} | {s['reused']} "
            f"| {s['llm_band_accuracy']:.3f} | {s['precision']:.3f} | {s['recall']:.3f} "
            f"| {s['f1']:.3f} | {s['missed_pos']} | {s['false_merges']} "
            f"| {s['sweep_f1']:.3f} @ {s['sweep_thr']:.2f} | {s['roc_auc']:.3f} |")
    bands = sorted({(s["band_lo"], s["band_hi"]) for s in stats.values()})
    band_txt = ", ".join(f"[{lo}, {hi}]" for lo, hi in bands)
    lines += ["", f"Band(s) = {band_txt}; composite: in-band -> 0.98/0.02 by LLM verdict, "
              "outside -> router score.", ""]
    open(md_path, "w").write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/company_pairs.csv")
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--router", required=True,
                    help="tfidf | alias | lt_comp_en | lt_comp_multi (see llm_matcher.build_blocker)")
    ap.add_argument("--decisions", nargs="+", required=True,
                    help="decision CSV(s) with pair_id + llm_same columns")
    ap.add_argument("--name", required=True,
                    help="short label: column/row becomes hybrid_<name>_llm")
    ap.add_argument("--band-lo", type=float, default=0.10)
    ap.add_argument("--band-hi", type=float, default=0.95)
    a = ap.parse_args()

    rows, pairs, y, ptype = load_pairs(a.pairs)
    ptype = np.array(ptype)

    router = build_blocker(a.router)
    rs = np.asarray(router.score_pairs(pairs), dtype=float)
    decisions = load_reuse(a.decisions)
    comp, in_band, missing = composite_scores(rows, rs, decisions, a.band_lo, a.band_hi)

    colname = f"hybrid_{a.name}_llm"
    pred = comp >= 0.5
    P, R, F, tp, fp, fn = prf(y, pred)
    fp_hard = int(np.sum(pred & (y == 0) & (ptype == "hard_negative")))
    fp_easy = int(np.sum(pred & (y == 0) & (ptype == "easy_negative")))

    band_ids = [i for i in range(len(rows)) if in_band[i] and rows[i]["pair_id"] in decisions]
    band_ok = sum(1 for i in band_ids if int(decisions[rows[i]["pair_id"]]) == y[i])
    band_acc = band_ok / len(band_ids) if band_ids else float("nan")

    (thr_s, P_s, R_s, F_s), _ = best_threshold(y, comp)
    auc, ap_ = roc_auc(y, comp)

    print(f"{colname}: band {int(in_band.sum())} (decided {len(band_ids)}, missing {missing}) · "
          f"LLM band accuracy {band_acc:.3f}")
    print(f"  fixed band [{a.band_lo},{a.band_hi}]: P={P:.3f} R={R:.3f} F1={F:.3f} "
          f"FP={fp} FN={fn}")
    print(f"  best-F1 sweep: F1={F_s:.3f} @ thr={thr_s:.2f} · ROC-AUC={auc:.3f}")

    sp = os.path.join(a.outdir, "pair_scores.csv")
    mp = os.path.join(a.outdir, "metrics.csv")
    upsert_column(sp, [r["pair_id"] for r in rows], colname, comp)
    upsert_row(mp, dict(matcher=colname, threshold=0.5,
                        precision=round(P, 4), recall=round(R, 4), f1=round(F, 4),
                        roc_auc=round(auc, 4), avg_precision=round(ap_, 4),
                        false_merge_hard=fp_hard, false_merge_easy=fp_easy, missed_pos=fn))

    stats_path = os.path.join(a.outdir, "hybrid_stats.json")
    stats = json.load(open(stats_path)) if os.path.exists(stats_path) else {}
    stats[colname] = dict(router=a.router, band_lo=a.band_lo, band_hi=a.band_hi,
                          band_size=int(in_band.sum()), reused=len(band_ids),
                          missing=missing, llm_band_accuracy=round(band_acc, 4),
                          precision=round(P, 4), recall=round(R, 4), f1=round(F, 4),
                          false_merges=fp, missed_pos=fn,
                          sweep_thr=round(thr_s, 4), sweep_f1=round(F_s, 4),
                          roc_auc=round(auc, 4), avg_precision=round(ap_, 4))
    json.dump(stats, open(stats_path, "w"), indent=2)
    render_comparison(stats_path, os.path.join(a.outdir, "router_comparison.md"))
    print(f"updated {sp}, {mp}, {stats_path}, router_comparison.md")


if __name__ == "__main__":
    main()
