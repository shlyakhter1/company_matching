#!/usr/bin/env python3
"""
score_holdings.py — grade a match_holdings.py run against a hand-labeled answer key.

match_holdings.py has no ground truth of its own; this compares its output
(out/fund_holdings_matched.csv) against data/fund_holdings_ground_truth.csv
(company_name -> expected_canonical, blank = correctly expected to have no match).

Usage:
  python score_holdings.py
  python score_holdings.py --matched out/fund_holdings_matched.csv --truth data/fund_holdings_ground_truth.csv
"""
import csv, argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matched", default="out/fund_holdings_matched.csv")
    ap.add_argument("--truth", default="data/fund_holdings_ground_truth.csv")
    a = ap.parse_args()

    truth = {r["company_name"]: r["expected_canonical"] for r in csv.DictReader(open(a.truth))}
    rows = list(csv.DictReader(open(a.matched)))

    tp = fp = fn = tn = 0
    errors = []
    for r in rows:
        name = r["company_name"]
        expected = truth.get(name, "")
        got = r["matched_canonical"]
        if expected:
            if got == expected:
                tp += 1
            elif got:
                fp += 1
                errors.append(("false_merge", name, expected, got, r["match_score"]))
            else:
                fn += 1
                errors.append(("missed", name, expected, "", r["match_score"]))
        else:
            if got:
                fp += 1
                errors.append(("spurious_match", name, "(none)", got, r["match_score"]))
            else:
                tn += 1

    n = tp + fp + fn + tn
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else float("nan")
    accuracy = (tp + tn) / n if n else float("nan")

    print(f"n={n}  TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"precision={precision:.3f}  recall={recall:.3f}  f1={f1:.3f}  accuracy={accuracy:.3f}")

    if errors:
        print(f"\n{'type':<16} {'company_name':<32} {'expected':<38} {'got':<38} score")
        print("-" * 130)
        for kind, name, expected, got, score in errors:
            print(f"{kind:<16} {name:<32} {expected:<38} {got:<38} {score}")

if __name__ == "__main__":
    main()
