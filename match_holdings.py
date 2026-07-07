#!/usr/bin/env python3
"""
match_holdings.py — apply the lt-wikidata-comp-en model to fund holdings.

For each holding, scores it against all canonical names in a reference registry
and returns the best match above a confidence threshold.

Usage:
  python match_holdings.py
  python match_holdings.py --holdings data/fund_holdings.csv --threshold 0.7
  python match_holdings.py --registry data/company_records.csv --outdir out
"""
import argparse, csv, os
import numpy as np
from matchers import SentenceTransformerMatcher

MATCHER_ID = "dell-research-harvard/lt-wikidata-comp-en"

def load_registry(path):
    """Return list of (name_variant, canonical_name) for every row — tickers,
    typos, and former names included, so a query can match any of them."""
    with open(path) as f:
        return [(row["name_variant"], row["canonical_name"]) for row in csv.DictReader(f)]

def load_holdings(path):
    with open(path) as f:
        return list(csv.DictReader(f))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdings",  default="data/fund_holdings.csv")
    ap.add_argument("--registry",  default="data/company_records.csv")
    ap.add_argument("--outdir",    default="out")
    ap.add_argument("--threshold", type=float, default=0.7,
                    help="Min score to emit a match (default 0.7)")
    a = ap.parse_args()

    print(f"Loading matcher: {MATCHER_ID}")
    matcher = SentenceTransformerMatcher(MATCHER_ID, short="lt_comp_en")

    registry   = load_registry(a.registry)
    variants   = [v for v, c in registry]
    canonicals = [c for v, c in registry]
    holdings   = load_holdings(a.holdings)
    print(f"Registry: {len(variants)} name variants across {len(set(canonicals))} companies")
    print(f"Holdings: {len(holdings)} rows")

    # Embed every variant once (tickers, typos, former names — not just canonicals)
    print("Embedding registry...")
    canon_vecs = matcher.embed(variants)

    results = []
    print("Matching holdings...")
    for h in holdings:
        query = h["company_name"]
        q_vec = matcher.embed([query])
        # cosine similarity against all canonicals
        c = canon_vecs / (np.linalg.norm(canon_vecs, axis=1, keepdims=True) + 1e-9)
        q = q_vec   / (np.linalg.norm(q_vec,   axis=1, keepdims=True) + 1e-9)
        scores = (c @ q.T).flatten()
        best_idx   = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        results.append({
            "company_name":      query,
            "fund_name":         h["fund_name"],
            "holding_value":     h["holding_value"],
            "matched_canonical": canonicals[best_idx] if best_score >= a.threshold else "",
            "match_score":       f"{best_score:.4f}",
            "match_status":      "matched" if best_score >= a.threshold else "no_match",
        })

    os.makedirs(a.outdir, exist_ok=True)
    outpath = os.path.join(a.outdir, "fund_holdings_matched.csv")
    fieldnames = ["company_name","fund_name","holding_value",
                  "matched_canonical","match_score","match_status"]
    with open(outpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)

    matched = sum(1 for r in results if r["match_status"] == "matched")
    print(f"\nDone. {matched}/{len(results)} holdings matched (threshold={a.threshold})")
    print(f"Output: {outpath}")

    # Print a quick summary table
    print(f"\n{'Company name':<40} {'Matched canonical':<40} {'Score':>6} {'Status'}")
    print("-" * 100)
    for r in results:
        print(f"{r['company_name']:<40} {r['matched_canonical']:<40} {r['match_score']:>6} {r['match_status']}")

if __name__ == "__main__":
    main()
