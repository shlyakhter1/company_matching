# CLAUDE_CODE_INSTRUCTIONS.md — Round 4: Embedding Routers + Standard Benchmark

Task spec for Claude Code working in this repository (company_matching). Execute
requirements in order; each has acceptance criteria. Do not begin a REQ until the
previous REQ's acceptance criteria pass.

## Context (read first)

This repo benchmarks company-name matching with a hybrid architecture:
**router (cheap scorer) → dual thresholds → LLM adjudication of the uncertain
band → metrics & visuals.** Rounds so far, on the 955-pair hand-built dataset
(fixed band [0.10, 0.95]):

| pipeline | P | R | F1 | misses |
|---|---|---|---|---|
| Hybrid v1 — tfidf router | 0.987 | 0.647 | 0.782 | 165 |
| Hybrid v2 — alias router (`AliasRouter` in `matchers.py`) | 0.989 | 0.754 | 0.856 | 115 |

The LLM adjudicator has been perfect so far (438/438 band pairs, then 488/488);
**all residual errors belong to the router.** The 115 remaining misses are
knowledge-only cases (中国银行↔Bank of China, Fiat Chrysler→Stellantis,
Daimler→Mercedes-Benz, initialisms broken by normalization like HSBC). Round 4
attacks them with two name-specialized embedding routers and adds a standard
benchmark for external comparability.

Key files: `matchers.py` (matcher registry), `benchmark.py`, `visualize.py`,
`llm_matcher.py` (LLM stage; `--blocker` selects the router),
`load_opensanctions_pairs.py` (converter), `data/company_records.csv` +
`data/company_pairs.csv` (hand-built dataset), `out/llm_band2_decisions.csv`
(existing adjudications — 488 pairs).

## Hard constraints

- **C1 — Keep the existing dataset.** Do not modify, regenerate, or delete
  `data/company_records.csv`, `data/company_pairs.csv`, or `make_dataset.py`'s
  DROP/CLUSTERS structures. It stays the primary eyeball-able dataset; the new
  dataset is additive.
- **C2 — Do not modify existing decision files.** `out/llm_band*_decisions.csv`
  are frozen experiment records. New adjudications go to new files
  (`out/llm_band3_decisions.csv`, `out/llm_band4_decisions.csv`, `out_os/...`).
- **C3 — Preserve routing semantics.** Fixed band stays [0.10, 0.95] for
  comparability with v1/v2. Report best-F1 sweeps *in addition*, never instead.
- **C4 — Reuse, don't re-adjudicate.** For pairs already decided in
  `out/llm_band2_decisions.csv`, reuse the stored decision; send the LLM only
  pairs newly entering a band.
- **C5 — Report failures honestly.** If a model download fails or a router
  underperforms, record the result; do not silently substitute.

## Requirements

### REQ-1 — Environment
Install `sentence-transformers` and `torch` (CPU is fine). Set
`ANTHROPIC_API_KEY` (or `OPENAI_API_KEY` with `--backend openai`) for the LLM
stage.
**Accept:** `python matchers.py` lists `st_lt_comp_en` and `st_eridu` as OK.

### REQ-2 — Router A: LinkTransformer
Add `EmbeddingRouter` to `matchers.py`: wraps a `SentenceTransformerMatcher`
and exposes the same interface as `AliasRouter` (`score_pairs`, `embed`,
`.name`). Instantiate with `dell-research-harvard/lt-wikidata-comp-en`,
`name="router_lt_comp_en"`. Add to `default_registry()`. Optionally combine
with alias rules: `score = max(embedding_cos, alias_rules)` behind a flag
`with_alias=True` (default True) — rules are free recall, keep them.
**Accept:** `EmbeddingRouter` scores `("Goldman Sachs","GS")` and
`("Amazon.com, Inc.","AMZN")` above 0.5 and
`("General Electric","General Mills")` below the pair's alias-rule floor.

### REQ-3 — Router B: eridu
Same wrapper, `Graphlet-AI/eridu`, `name="router_eridu"`. Note eridu outputs
similarity for person/company names cross-script; verify its score range and
rescale to [0,1] if needed (document the rescaling in a comment).
**Accept:** `("Газпром","Gazprom PJSC")` and `("中国银行","Bank of China")`
score above the band floor (≥0.10) — the exact cases Round 3 could not reach.

### REQ-4 — Hybrid v3 & v4 on the hand-built dataset
For each router (v3 = lt-comp-en, v4 = eridu):
1. Score all 955 pairs; compute the new band at [0.10, 0.95].
2. Route to the LLM only pairs not present in `out/llm_band2_decisions.csv`
   (per C4) via `llm_matcher.py pairs --blocker <router>`; write
   `out/llm_band3_decisions.csv` / `out/llm_band4_decisions.csv`.
3. Compute fixed-band metrics (P/R/F1, band size, FP/FN) and the best-F1 sweep
   + ROC-AUC on the composite score (band → 0.98/0.02 by LLM verdict, outside →
   router score). Append `hybrid_ltcomp_llm` and `hybrid_eridu_llm` columns to
   `out/pair_scores.csv` and rows to `out/metrics.csv`, matching the existing
   `hybrid_alias_llm` conventions exactly.
**Accept:** `out/metrics.csv` contains both new rows; a comparison table
v1→v4 is printed and saved to `out/router_comparison.md`. Expected direction:
misses < 115 and auto-accepted Merck false-merges resolved (embedding scores
for Merck&Co↔MerckKGaA should fall inside the band, where the LLM catches
them) — if not, report per C5.

### REQ-5 — Add the standard dataset (OpenSanctions Pairs)
1. Download: `curl -o pairs.json
   https://data.opensanctions.org/contrib/training/pairs.json` (if moved, see
   https://www.opensanctions.org/docs/opensource/pairs/ ; the dataset is CC-BY,
   attribute OpenSanctions).
2. Convert with the existing loader — do not rewrite it:
   `python load_opensanctions_pairs.py --pairs pairs.json --out
   data/os_pairs_5k.csv --schemas Company Organization LegalEntity --sample
   5000 --seed 42`. Keep `--seed 42` so the sample is reproducible.
3. Also emit a small fixed dev slice: `--sample 600 --seed 7 --out
   data/os_pairs_dev.csv` for cheap iteration.
**Accept:** both CSVs exist, schema identical to `data/company_pairs.csv`
(7 columns), positives ≈ negatives, and `data/` still contains the untouched
hand-built files (C1).

### REQ-6 — Benchmark both datasets
Run `benchmark.py` and the four hybrids (tfidf, alias, lt-comp-en, eridu
routers + LLM band) on `data/os_pairs_5k.csv`, outputs to `out_os/`. Use
`--band-lo/--band-hi` defaults; cap LLM spend by batching (default batch=25)
and report the number of LLM calls. Produce `out_os/metrics.csv` and
regenerate visuals (`visualize.py --outdir out_os --records` — records file is
not available for OS pairs, so skip the embedding map with a flag; add
`--no-map` to visualize.py if needed, minimal diff).
**Accept:** `out_os/metrics.csv` has ≥6 rows; a summary compares our best
pipeline against the published baselines (rule-based 91.3 F1, GPT-4o 98.95,
DeepSeek-R1-14B 98.23 — from arXiv:2603.11051) with the caveat that our
5K sample ≠ their full-set evaluation protocol.

### REQ-7 — Consolidate
Write `out/ROUND4_REPORT.md`: router comparison table (v1–v4, both datasets),
LLM call counts and estimated cost, the Merck-false-merge status, remaining
misses with examples, and a recommendation (which router to default). Update
README.md's matcher table with the two routers (3-line diff max).
**Accept:** report exists; `python run_all.py` still passes end-to-end on the
hand-built dataset (regression check).

## Out of scope (do not do)
- Fine-tuning any model; changing band thresholds as the primary result;
  deleting or "cleaning" the hand-built dataset; adjudicating with heuristics
  instead of the LLM; committing `pairs.json` (add to .gitignore — it is large).
