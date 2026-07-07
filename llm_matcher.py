#!/usr/bin/env python3
"""
llm_matcher.py — run company-name matching as a separate LLM task.

Two modes, mirroring the existing tasks:

  holdings  For each holding, a cheap blocker (TF-IDF by default) retrieves the
            top-k registry candidates, then the LLM adjudicates: pick one
            canonical or NONE. Output = out/fund_holdings_matched_llm.csv, in the
            exact schema of match_holdings.py, so score_holdings.py grades it
            unchanged.

  pairs     Classify labeled pairs from data/company_pairs.csv as same-company /
            different. By default only the UNCERTAIN MIDDLE BAND of a baseline
            matcher is sent to the LLM (the layered architecture); --all sends
            every pair. Output = out/llm_pair_decisions.csv.

Backends: Anthropic (default; needs ANTHROPIC_API_KEY) or OpenAI (--backend
openai; needs OPENAI_API_KEY). Requests are batched: many items per prompt,
strict-JSON responses, with one retry on parse failure.

Usage:
  python llm_matcher.py holdings
  python llm_matcher.py holdings --top-k 5 --model claude-sonnet-4-6
  python llm_matcher.py pairs --band-lo 0.10 --band-hi 0.95
  python llm_matcher.py pairs --blocker lt_comp_en --reuse out/llm_band2_decisions.csv \
      --out-name llm_band3_decisions.csv
  python llm_matcher.py pairs --all --backend openai --model gpt-4o-mini
"""
import argparse, csv, json, os, re, sys, time
import numpy as np

# --------------------------------------------------------------------------
# prompts
# --------------------------------------------------------------------------
SYS = ("You are an entity-resolution adjudicator for company names. Decide whether "
       "names refer to the SAME legal company. Rules: tickers, acronyms, typos, "
       "transliterations, and FORMER names of the same company count as the same "
       "(Facebook=Meta, Daimler=Mercedes-Benz Group, Royal Dutch Shell=Shell). "
       "Different companies that share words are NOT the same (Merck & Co. vs Merck KGaA; "
       "HP Inc. vs Hewlett Packard Enterprise; United Airlines vs United Parcel Service; "
       "parent vs subsidiary like Alphabet vs Google are different unless told otherwise). "
       "Reply ONLY with the JSON asked for — no prose, no markdown fences.")

HOLDINGS_ITEM = ("Item {i}: query name: \"{q}\"\n  candidates:\n{cands}\n")

HOLDINGS_TASK = ("For each item pick the candidate canonical name that is the SAME company as "
                 "the query, or null if none is. Respond with a JSON array like "
                 '[{{"i": 1, "match": "<canonical or null>", "confidence": 0.0-1.0}}, ...] '
                 "covering every item.\n\n{items}")

PAIRS_TASK = ("For each numbered pair say whether the two names refer to the same company. "
              'Respond with a JSON array like [{{"i": 1, "same": true/false, '
              '"confidence": 0.0-1.0}}, ...] covering every pair.\n\n{items}')


# --------------------------------------------------------------------------
# LLM backends
# --------------------------------------------------------------------------
def call_anthropic(model, system, user, max_tokens=4000):
    import anthropic
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    r = client.messages.create(model=model, max_tokens=max_tokens,
                               system=system,
                               messages=[{"role": "user", "content": user}])
    return "".join(b.text for b in r.content if b.type == "text")

def call_openai(model, system, user, max_tokens=4000):
    from openai import OpenAI
    client = OpenAI()  # reads OPENAI_API_KEY
    r = client.chat.completions.create(model=model, max_tokens=max_tokens,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}])
    return r.choices[0].message.content

def llm(backend, model, system, user, retries=2):
    fn = call_anthropic if backend == "anthropic" else call_openai
    last = None
    for attempt in range(retries + 1):
        try:
            txt = fn(model, system, user)
            txt = re.sub(r"^```(json)?|```$", "", txt.strip(), flags=re.M).strip()
            return json.loads(txt)
        except Exception as e:
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"LLM call failed after retries: {last}")


# --------------------------------------------------------------------------
# blocking (candidate retrieval) — TF-IDF by default, embeddings if available
# --------------------------------------------------------------------------
def build_blocker(kind):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import matchers as M
    return M.build_blocker(kind)   # one registry for all tools; typos fail loudly

def topk_candidates(blocker, queries, variants, canonicals, k):
    """Return, per query, the top-k (variant, canonical, score) by cosine."""
    all_names = list(variants) + list(queries)
    X = blocker.embed(all_names)
    X = np.asarray(X, dtype=float)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    V, Q = X[: len(variants)], X[len(variants):]
    sims = Q @ V.T                                     # [nq, nv]
    out = []
    for row in sims:
        idx = np.argsort(-row)[: max(k * 3, k)]        # over-fetch, dedupe canonicals
        seen, cands = set(), []
        for j in idx:
            c = canonicals[j]
            if c in seen:
                continue
            seen.add(c)
            cands.append((variants[j], c, float(row[j])))
            if len(cands) == k:
                break
        out.append(cands)
    return out


# --------------------------------------------------------------------------
# mode: holdings
# --------------------------------------------------------------------------
def run_holdings(a):
    registry = list(csv.DictReader(open(a.registry, encoding="utf-8")))
    variants  = [r["name_variant"] for r in registry]
    canonicals = [r["canonical_name"] for r in registry]
    holdings = list(csv.DictReader(open(a.holdings, encoding="utf-8")))
    queries = [h["company_name"] for h in holdings]
    print(f"registry: {len(variants)} variants / {len(set(canonicals))} companies · holdings: {len(queries)}")

    blocker = build_blocker(a.blocker)
    cands = topk_candidates(blocker, queries, variants, canonicals, a.top_k)

    decisions = {}
    for start in range(0, len(queries), a.batch):
        chunk = list(range(start, min(start + a.batch, len(queries))))
        items = "".join(
            HOLDINGS_ITEM.format(
                i=i + 1, q=queries[i],
                cands="".join(f"    - \"{c}\"  (registry variant matched: \"{v}\", blocker score {s:.2f})\n"
                              for v, c, s in cands[i]))
            for i in chunk)
        resp = llm(a.backend, a.model, SYS, HOLDINGS_TASK.format(items=items))
        for d in resp:
            decisions[chunk[d["i"] - 1]] = (d.get("match"), float(d.get("confidence", 0.5)))
        print(f"  adjudicated {min(start + a.batch, len(queries))}/{len(queries)}")

    os.makedirs(a.outdir, exist_ok=True)
    outp = os.path.join(a.outdir, "fund_holdings_matched_llm.csv")
    with open(outp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["company_name", "fund_name", "holding_value",
                                          "matched_canonical", "match_score", "match_status"])
        w.writeheader()
        for i, h in enumerate(holdings):
            match, conf = decisions.get(i, (None, 0.0))
            matched = match if (match and conf >= a.threshold) else ""
            w.writerow({"company_name": h["company_name"], "fund_name": h["fund_name"],
                        "holding_value": h["holding_value"],
                        "matched_canonical": matched, "match_score": f"{conf:.4f}",
                        "match_status": "matched" if matched else "no_match"})
    print(f"wrote {outp}  (score it with: python score_holdings.py --matched {outp})")


# --------------------------------------------------------------------------
# mode: pairs
# --------------------------------------------------------------------------
def append_partial(path, items):
    """Append (pair_id, llm_same) decisions after each batch so an interrupted
    run loses at most one batch; run_pairs auto-reuses the file on restart."""
    fresh = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if fresh:
            w.writerow(["pair_id", "llm_same"])
        w.writerows(items)


def load_reuse(paths):
    """Prior decisions keyed by pair_id, from any CSV with pair_id + llm_same columns
    (both the frozen llm_band*_decisions.csv records and this script's own output)."""
    reuse = {}
    for p in paths:
        for r in csv.DictReader(open(p, encoding="utf-8")):
            v = (r.get("llm_same") or "").strip()
            if v in ("0", "1", "True", "False"):
                reuse[r["pair_id"]] = int(v in ("1", "True"))
    return reuse


def run_pairs(a):
    from benchmark import load_pairs, prf
    rows, pairs, y, _ptype = load_pairs(a.pairs)
    print(f"pairs: {len(rows)}")

    if a.all:
        band = list(range(len(rows)))
        base_scores = None
    else:
        blocker = build_blocker(a.blocker)
        s = np.asarray(blocker.score_pairs(pairs), float)
        base_scores = s
        band = [i for i in range(len(rows)) if a.band_lo <= s[i] <= a.band_hi]
        print(f"baseline {a.blocker}: auto-reject {int(np.sum(s < a.band_lo))} · "
              f"auto-accept {int(np.sum(s > a.band_hi))} · LLM band {len(band)}")

    os.makedirs(a.outdir, exist_ok=True)
    outp = os.path.join(a.outdir, a.out_name)
    partial = outp + ".partial"   # per-batch persistence; survives a crashed run
    reuse_files = list(a.reuse or [])
    if os.path.exists(partial):
        print(f"resuming: reusing decisions from interrupted run ({partial})")
        reuse_files.append(partial)
    reuse = load_reuse(reuse_files) if reuse_files else {}
    decisions = {}   # row idx -> (same, confidence, source)
    if reuse:
        for i in band:
            pid = rows[i]["pair_id"]
            if pid in reuse:
                decisions[i] = (bool(reuse[pid]), "", "reused")
        print(f"reused {len(decisions)}/{len(band)} band decisions (C4); "
              f"{len(band) - len(decisions)} new pairs go to the LLM")
    band = [i for i in band if i not in decisions]

    n_calls = 0

    def ask(idxs):
        """One adjudication call; returns {row idx: (same, conf)} for the items it covered."""
        nonlocal n_calls
        items = "".join(f'Pair {j + 1}: "{rows[i]["name_a"]}"  vs  "{rows[i]["name_b"]}"\n'
                        for j, i in enumerate(idxs))
        resp = llm(a.backend, a.model, SYS, PAIRS_TASK.format(items=items))
        n_calls += 1
        out = {}
        for d in resp:
            j = int(d.get("i", 0))
            if 1 <= j <= len(idxs):
                out[idxs[j - 1]] = (bool(d["same"]), float(d.get("confidence", 0.5)))
        return out

    for start in range(0, len(band), a.batch):
        chunk = band[start: start + a.batch]
        got = ask(chunk)
        missing = [i for i in chunk if i not in got]
        if missing:   # the model dropped items from its JSON; re-ask just those once
            got.update(ask(missing))
            missing = [i for i in chunk if i not in got]
        if missing:
            raise RuntimeError(
                f"LLM response still missing {len(missing)} pairs after retry "
                f"(e.g. {rows[missing[0]]['pair_id']}); progress saved to {partial} — rerun to resume")
        for i, (same, conf) in got.items():
            decisions[i] = (same, conf, "new")
        append_partial(partial, [(rows[i]["pair_id"], int(got[i][0])) for i in chunk])
        print(f"  adjudicated {min(start + a.batch, len(band))}/{len(band)}")
    print(f"LLM calls: {n_calls}")
    with open(outp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "name_a", "name_b", "label", "pair_type",
                    "route", "baseline_score", "llm_same", "llm_confidence", "final_pred",
                    "source"])
        for i, r in enumerate(rows):
            if i in decisions:
                same, conf, src = decisions[i]
                route, pred = "llm", int(same)
                same = int(same)
            elif base_scores is not None:
                route = "auto_accept" if base_scores[i] > a.band_hi else "auto_reject"
                same, conf, pred, src = "", "", int(base_scores[i] > a.band_hi), ""
            else:
                route, same, conf, pred, src = "skipped", "", "", 0, ""
            w.writerow([r["pair_id"], r["name_a"], r["name_b"], r["label"], r["pair_type"],
                        route, "" if base_scores is None else f"{base_scores[i]:.4f}",
                        same, conf, pred, src])
    if os.path.exists(partial):
        os.remove(partial)   # complete run recorded in outp; the crash file is obsolete
    # quick metrics
    pred = np.array([int(r["final_pred"]) for r in csv.DictReader(open(outp))])
    P, R, F, *_ = prf(y, pred == 1)
    print(f"pipeline metrics: P={P:.3f} R={R:.3f} F1={F:.3f}")
    print(f"wrote {outp}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--backend", choices=["anthropic", "openai"], default="anthropic")
    common.add_argument("--model", default="claude-sonnet-4-6")
    common.add_argument("--blocker", default="tfidf",
                        help="'tfidf', 'alias' (tfidf + initialism/ticker rules), or a sentence-transformers model id")
    common.add_argument("--batch", type=int, default=25, help="items per LLM call")
    common.add_argument("--outdir", default="out")

    h = sub.add_parser("holdings", parents=[common])
    h.add_argument("--holdings", default="data/fund_holdings.csv")
    h.add_argument("--registry", default="data/company_records.csv")
    h.add_argument("--top-k", type=int, default=5)
    h.add_argument("--threshold", type=float, default=0.5,
                   help="min LLM confidence to emit a match")

    p = sub.add_parser("pairs", parents=[common])
    p.add_argument("--pairs", default="data/company_pairs.csv")
    p.add_argument("--all", action="store_true", help="send every pair to the LLM")
    p.add_argument("--band-lo", type=float, default=0.10,
                   help="baseline score below which pairs auto-reject")
    p.add_argument("--band-hi", type=float, default=0.95,
                   help="baseline score above which pairs auto-accept")
    p.add_argument("--reuse", nargs="+", default=None, metavar="CSV",
                   help="prior decision files (pair_id + llm_same columns); band pairs "
                        "already decided there are reused instead of re-adjudicated (C4)")
    p.add_argument("--out-name", default="llm_pair_decisions.csv",
                   help="output filename within --outdir")

    a = ap.parse_args()
    key = "ANTHROPIC_API_KEY" if a.backend == "anthropic" else "OPENAI_API_KEY"
    if key not in os.environ:
        sys.exit(f"error: {key} not set")
    (run_holdings if a.mode == "holdings" else run_pairs)(a)


if __name__ == "__main__":
    main()
