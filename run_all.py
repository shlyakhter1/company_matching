#!/usr/bin/env python3
"""
run_all.py — regenerate data (optional), benchmark all matchers, render all visuals.

Examples:
  python run_all.py                       # use existing data, run everything available
  python run_all.py --regen               # rebuild the ~120-cluster dataset first
  python run_all.py --regen --keep-all    # rebuild with the full ~159-cluster set
  python run_all.py --matchers fuzzy_token_sort_norm tfidf_char
  python run_all.py --embed-matcher st_bge_m3 --reducer umap --focus st_bge_m3
"""
import argparse, subprocess, sys, os

HERE = os.path.dirname(os.path.abspath(__file__))

def run(cmd):
    print("\n$ " + " ".join(cmd))
    subprocess.run([sys.executable] + cmd, cwd=HERE, check=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regen", action="store_true")
    ap.add_argument("--keep-all", action="store_true")
    ap.add_argument("--max-clusters", type=int, default=None)
    ap.add_argument("--matchers", nargs="*", default=None)
    ap.add_argument("--focus", default=None)
    ap.add_argument("--embed-matcher", default="tfidf_char")
    ap.add_argument("--reducer", default="umap")
    a = ap.parse_args()

    if a.regen:
        cmd = ["make_dataset.py", "--outdir", "data"]
        if a.keep_all: cmd.append("--keep-all")
        if a.max_clusters: cmd += ["--max-clusters", str(a.max_clusters)]
        run(cmd)

    bcmd = ["benchmark.py", "--records", "data/company_records.csv",
            "--pairs", "data/company_pairs.csv", "--outdir", "out"]
    if a.matchers: bcmd += ["--matchers"] + a.matchers
    run(bcmd)

    vcmd = ["visualize.py", "--outdir", "out", "--records", "data/company_records.csv",
            "--embed-matcher", a.embed_matcher, "--reducer", a.reducer]
    if a.focus: vcmd += ["--focus", a.focus]
    run(vcmd)

    print("\nAll done. See ./out for metrics.csv, pair_scores.csv, *.png and *.html")

if __name__ == "__main__":
    main()
