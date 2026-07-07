#!/usr/bin/env python3
"""
load_opensanctions_pairs.py — convert the OpenSanctions Pairs benchmark into
this repo's company_pairs.csv schema, so benchmark.py / visualize.py /
llm_matcher.py run on it unchanged.

Why this dataset: it is the standard, published benchmark closest to our task —
755K analyst-labeled entity pairs (48K companies, 47K organizations among 1M
entities), multilingual and cross-script, with published baselines to compare
against (arXiv:2603.11051 — rule-based nomenklatura 91.3 F1; GPT-4o 98.95 F1;
DeepSeek-R1-Distill-Qwen-14B 98.23 F1).

Get the data (not fetchable from a sandbox; run on your machine):
  curl -o pairs.json https://data.opensanctions.org/contrib/training/pairs.json
  # if the location has moved, see https://www.opensanctions.org/docs/opensource/pairs/

Then:
  python load_opensanctions_pairs.py --pairs pairs.json --out data/os_pairs.csv \
      --schemas Company Organization LegalEntity --sample 5000 --seed 42
  python benchmark.py --pairs data/os_pairs.csv --outdir out_os
  python llm_matcher.py pairs --pairs data/os_pairs.csv --blocker tfidf --outdir out_os

Each input line is a JSON object with 'left' and 'right' FtM entities and a
'judgement' of positive/negative/unsure ('unsure' rows are skipped).
"""
import argparse, csv, json, random, sys

def primary_name(ent):
    props = ent.get("properties", {}) or {}
    for key in ("name", "alias", "weakAlias"):
        vals = props.get(key) or []
        if vals:
            return str(vals[0]).strip()
    return str(ent.get("caption", "")).strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="pairs.json from OpenSanctions")
    ap.add_argument("--out", default="data/os_pairs.csv")
    ap.add_argument("--schemas", nargs="*", default=["Company", "Organization", "LegalEntity"],
                    help="entity schemas to keep (use ['Person'] for people)")
    ap.add_argument("--sample", type=int, default=None,
                    help="stratified sample size (balanced pos/neg); default = all")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    keep = set(a.schemas)
    pos, neg = [], []
    skipped = 0
    with open(a.pairs, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1; continue
            j = row.get("judgement") or row.get("judgment")
            if j not in ("positive", "negative"):
                continue
            l, r = row.get("left", {}), row.get("right", {})
            if l.get("schema") not in keep or r.get("schema") not in keep:
                continue
            na, nb = primary_name(l), primary_name(r)
            if not na or not nb:
                skipped += 1; continue
            (pos if j == "positive" else neg).append((na, nb))

    rng = random.Random(a.seed)
    if a.sample:
        k = a.sample // 2
        pos = rng.sample(pos, min(k, len(pos)))
        neg = rng.sample(neg, min(k, len(neg)))
    pairs = [(na, nb, 1, "os_positive") for na, nb in pos] + \
            [(na, nb, 0, "os_negative") for na, nb in neg]
    rng.shuffle(pairs)

    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "name_a", "name_b", "label", "pair_type", "cluster_a", "cluster_b"])
        for i, (na, nb, y, pt) in enumerate(pairs, 1):
            w.writerow([f"os{i:06d}", na, nb, y, pt, "", ""])

    print(f"kept {len(pos)} positive / {len(neg)} negative pairs "
          f"(schemas={sorted(keep)}, skipped={skipped})")
    print(f"wrote {a.out} — run: python benchmark.py --pairs {a.out} --outdir out_os")

if __name__ == "__main__":
    main()
