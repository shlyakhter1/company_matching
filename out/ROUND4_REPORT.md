# Round 4 Report — Embedding Routers + Standard Benchmark

Architecture under test: **router (cheap scorer) → fixed band [0.10, 0.95] →
LLM adjudication of the uncertain band**. Round 4 adds two name-specialized
embedding routers and the OpenSanctions Pairs standard benchmark.

## Deviations from the spec (C5 — reported, not hidden)

1. **Router B substitution.** `Graphlet-AI/eridu` has been removed from
   Hugging Face Hub (404 both anonymous and authenticated; no GitHub release
   artifacts; the author describes the published MiniLM-based version as
   obsolete). With the user's approval, Router B is
   `dell-research-harvard/lt-wikidata-comp-multi` — same LinkTransformer family
   as Router A but multilingual/cross-script, i.e. aimed at exactly the
   Газпром/中国银行 cases eridu was chosen for. Columns/rows are named
   `*ltmulti*`, not `*eridu*`, to avoid misattribution.
2. **Score mapping.** The inherited `(cos+1)/2` mapping compresses these
   contrastive name models into [0.5, 1.0]: Merck&Co↔MerckKGaA landed at 0.975
   (auto-accept — the exact false merge Round 4 was meant to catch) and nothing
   fell below 0.10. `EmbeddingRouter` therefore scores with **raw cosine clipped
   to [0,1]** (documented in `matchers.py`); the bare `st_*` matchers keep the
   old mapping for backward comparability of earlier columns.
3. **Adjudicator change.** Reused band2 decisions (488 pairs) were made by
   Claude (Fable 5) adjudicating in-context in Rounds 2–3. All *new* Round-4
   decisions come from **claude-sonnet-4-6 via the Anthropic API**
   (`llm_matcher.py`, batch 25). The perfect 926/926 streak of Rounds 2–3 did
   not survive contact with harder cases: band accuracy is now 0.993 (v3) /
   0.994 (v4) on the hand-built set and ~0.92 on OpenSanctions dev.
4. **OpenAI backend unavailable.** The configured `OPENAI_API_KEY` is out of
   quota (`insufficient_quota`), so `openai_small` FAILed in benchmarks and the
   OpenAI adjudication backend was not usable as a fallback.

## Hand-built dataset (955 pairs, fixed band [0.10, 0.95])

| pipeline | router | band | new LLM pairs | LLM calls | P | R | F1 | FP | misses |
|---|---|---|---|---|---|---|---|---|---|
| Hybrid v1 | tfidf_char | 438 | — (frozen) | — | 0.987 | 0.647 | 0.782 | 4 | 165 |
| Hybrid v2 | alias_router | 488 | — (frozen) | — | 0.989 | 0.754 | 0.856 | 4 | 115 |
| Hybrid v3 | router_lt_comp_en | 596 | 169 | 7 | 0.996 | 0.962 | 0.978 | 2 | 18 |
| **Hybrid v4** | **router_lt_comp_multi** | 697 | 90 (after v3 reuse) | 4 | **0.998** | **0.983** | **0.990** | **1** | **8** |

v1/v2 rows were recomputed by `hybrid_eval.py` from the frozen decision files
and reproduce the Round-2/3 summary exactly (machinery validation). Best-F1
sweeps: v3 0.978, v4 0.990, both at thr 0.95; ROC-AUC 0.976 / 0.990.

### Merck false-merge status

- v1/v2: 4 Merck&Co↔MerckKGaA variants auto-accepted at score 1.0.
- v3: 2 remain auto-accepted (`Merck and Co`↔`Merck Group` 0.982,
  `Merck & Co., Inc.`↔`Merck Group` 0.983); the other 2 fell into the band and
  the LLM rejected them. **Partial** resolution.
- v4: **all Merck pairs resolved** (in band → LLM rejects, or auto-reject).
  Its single remaining FP is `Hewlett Packard`↔`Hewlett Packard Enterprise
  Company` auto-accepted at >0.95 — a parent/spin-off hard negative.

### Remaining misses (v4, 8 total)

- **Below the band (auto-reject), 4:** ticker pairs with no name signal —
  `Salesforce`↔`CRM` ×3, `Facebook, Inc.`↔`META`. The initialism/contraction
  rules can't reach them (CRM is not a contraction of Salesforce) and the
  embedding places them near 0.
- **In band, LLM said "different", 5 (4 in v4):** `HSBC Holdings`↔`Hongkong and
  Shanghai Banking Corporation`, `Stellantis`↔`Fiat Chrysler Automobiles`,
  `Alphabet`↔`GOOGL`, `Hewlett Packard`↔`HPQ`. Several of these are
  label-vs-prompt-policy disagreements: the dataset labels parent/successor/
  holding relations as "same" while the adjudication prompt explicitly rules
  parent-vs-subsidiary "different" (e.g. Alphabet vs a Google ticker). Fixing
  this is a labeling-policy decision, not a router problem.

## OpenSanctions Pairs (5K balanced sample, seed 42)

Same fixed band, LLM decisions shared across routers via `--reuse` chaining
(3,975 unique band pairs adjudicated once instead of ~10,800 with independent
runs). Full table in `out_os/metrics.csv` / `out_os/router_comparison.md`.

| pipeline | band | new LLM pairs | LLM calls | P | R | F1 | ROC-AUC |
|---|---|---|---|---|---|---|---|
| hybrid_tfidf_llm | 1965 | 1965 | 79 | 0.838 | 0.662 | 0.740 | 0.695 |
| hybrid_alias_llm | 1968 | 3 | 1 | 0.837 | 0.663 | 0.740 | 0.695 |
| hybrid_ltcomp_llm | 3085 | 1331 | 54 | 0.835 | 0.873 | 0.854 | 0.852 |
| **hybrid_ltmulti_llm** | 3764 | 676 | 28 | 0.843 | **0.904** | **0.872** | 0.868 |

Router-only baselines on the same sample (best-F1 sweep, no LLM):
tfidf 0.667 · st_bge_m3 0.710 · lt-comp-multi 0.701 — the LLM band adds
+0.13–0.17 F1 over the best router alone.

### Comparison against published baselines — with heavy caveats

Published on this benchmark (arXiv:2603.11051, **full 755K set, full FtM
entity records**): rule-based nomenklatura **91.3** F1 · GPT-4o **98.95** ·
DeepSeek-R1-14B **98.23**. Our best pipeline reaches **87.2** on a 5K sample —
not directly comparable, and short of all three, for structural reasons worth
stating plainly:

- **Name-only input.** `load_opensanctions_pairs.py` keeps one primary name
  per entity and discards aliases, countries, registration numbers, dates —
  most of the signal the published systems consume. Some sampled negatives are
  literally identical strings (`"Company"` vs `"Company"`, label 0): no
  name-only system can score them correctly, which caps attainable F1 well
  below 100 on this conversion.
- **LLM band accuracy is ~0.89–0.90 here** (vs 0.99+ on the hand-built set):
  sanctions-list entities are obscure, multilingual, and often distinguishable
  only by non-name fields the adjudicator never sees.
- 5K balanced sample (seed 42) ≠ their full-set evaluation protocol.

The useful reading: the **relative** router ordering replicates the hand-built
result (multi > en > alias ≈ tfidf), and the hybrid architecture beats every
router-only baseline on both datasets.

## LLM usage & estimated cost (this round)

All adjudication with `claude-sonnet-4-6`, batch 25 pairs/call:

| run | new pairs | calls |
|---|---|---|
| hand-built v3 (lt-comp-en) | 169 | 7 |
| hand-built v4 (lt-comp-multi, after v3 reuse) | 90 | 4 |
| OS dev slice (pipeline validation) | 240 | 10 |
| OS 5K, tfidf | 1965 | 79 |
| OS 5K, alias (reuse tfidf) | 3 | 1 |
| OS 5K, lt-comp-en (reuse prior) | 1331 | 54 |
| OS 5K, lt-comp-multi (reuse prior) | 676 | 28 |
| **total** | **4474** | **183** |

At ~800 input + ~450 output tokens per call, ≈ 0.15M in / 0.08M out ≈
**$1.70 total** (Sonnet pricing) — the reuse chaining saved roughly 2.7× that.

## Recommendation

**Default router: `router_lt_comp_multi`** (lt-wikidata-comp-multi + alias-rule
floor, clipped-cosine scores). On the hand-built set it dominates every
alternative (F1 0.990, one FP, misses reduced 115→8) and its multilingual
training directly covers cross-script cases. `router_lt_comp_en` is a
reasonable English-only fallback with a slightly smaller band (596 vs 697,
i.e. ~15% fewer LLM calls at fixed band).

Caveats: the band grew with router quality (438→697 of 955 pairs), so LLM cost
per fully-new dataset rises; and the LLM is no longer a perfect oracle in the
band — residual errors are now split between router placement and adjudicator
judgment, mostly on genuinely ambiguous corporate-relationship pairs.
