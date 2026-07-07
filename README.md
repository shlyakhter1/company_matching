# Company-Name Matching Benchmark & Toolkit

A self-contained, re-runnable harness for experimenting with company-name entity
resolution. It ships a **~120-cluster labeled dataset**, a set of **pluggable
matchers** (fuzzy → TF-IDF → local embeddings → API embeddings), a **benchmark**
that scores them all, and **visualizations** that show the similarity scores and,
crucially, *where each matcher fails*.

The dataset is built so the easy cases are easy and the hard cases genuinely fight
back — confusable families like *United Airlines* vs *United Parcel Service*,
*Merck & Co.* (US) vs *Merck KGaA* (DE), *General Motors* vs *General Electric* vs
*General Mills*, and the *HP Inc.* vs *Hewlett Packard Enterprise* split.

---

## Quick start

```bash
pip install -r requirements.txt          # core deps run everything offline
python run_all.py --regen                # build data, benchmark, visualize
open out/embedding_map.html              # explore (or any file in ./out)
```

`run_all.py` with no `--regen` reuses the dataset already in `./data`.

### Enable the embedding & API matchers
They're already wired in — they just need their libraries (and, for APIs, keys):

```bash
pip install sentence-transformers torch        # local: bge-m3, lt-wikidata-comp, eridu
export OPENAI_API_KEY=sk-...                    # optional API matcher
export COHERE_API_KEY=...                       # optional API matcher
python run_all.py                              # they now light up automatically
```

Check what's active in your environment:

```bash
python matchers.py
```

---

## Files

| file | what it does |
|---|---|
| `make_dataset.py` | builds `data/company_records.csv` + `data/company_pairs.csv`. `--keep-all` for ~159 clusters, `--max-clusters N` to subset, `--seed` to reshuffle |
| `normalize.py` | shared normalization (legal-suffix stripping, `&`→and, accent folding) |
| `matchers.py` | the matcher backends behind one interface (see below) |
| `benchmark.py` | scores every available matcher → `out/pair_scores.csv`, `out/metrics.csv` |
| `visualize.py` | renders the four visualizations from the benchmark output |
| `run_all.py` | one command to do all of the above |

### Data schema
`company_records.csv` — one row per name variant: `record_id, cluster_id,
canonical_name, name_variant, country, variation_type`. Variants sharing a
`cluster_id` are the same company (the ground truth).

`company_pairs.csv` — labeled pairs: `pair_id, name_a, name_b, label,
pair_type, cluster_a, cluster_b`, where `pair_type ∈ {positive, hard_negative,
easy_negative}`.

---

## Matchers

| name | kind | runs offline? | notes |
|---|---|---|---|
| `fuzzy_token_sort` | RapidFuzz, raw | ✅ | baseline; no normalization |
| `fuzzy_token_sort_norm` | RapidFuzz + normalize | ✅ | shows how much normalization buys you |
| `fuzzy_token_set_norm` | RapidFuzz token-set | ✅ | robust to word order/extra tokens |
| `tfidf_char` | char n-gram TF-IDF cosine | ✅ | strong offline embedding baseline |
| `st_bge_m3` | `BAAI/bge-m3` | downloads | general dense embedding |
| `st_lt_comp_en` | `dell-research-harvard/lt-wikidata-comp-en` | downloads | **name-specialized** |
| `st_lt_comp_multi` | `dell-research-harvard/lt-wikidata-comp-multi` | downloads | **name-specialized**, multilingual (replaces `Graphlet-AI/eridu`, removed from HF Hub) |
| `router_lt_comp_en` | lt-comp-en + alias rules | downloads | Round-4 router: clipped cosine, alias-rule floor |
| `router_lt_comp_multi` | lt-comp-multi + alias rules | downloads | Round-4 router: best hybrid F1 0.990 with the LLM band |
| `openai_small` | `text-embedding-3-small` | API | needs `OPENAI_API_KEY` |
| `cohere_v4` | `embed-v4.0` | API | needs `COHERE_API_KEY` |

Add your own by subclassing the simple interface in `matchers.py`
(`.score_pairs(pairs) -> [0,1]` and optionally `.embed(names) -> vectors`).

---

## Visualizations (`./out`)

1. **`pr_curves.png`** — precision–recall for every matcher on one axes. Watch the
   lexical matchers collapse toward the random baseline as recall climbs past ~0.5:
   that's the hard negatives.
2. **`score_distributions.png`** — per-matcher histograms split into positive /
   hard-negative / easy-negative. The hard-negative mass overlapping the positives
   is the separability problem in one picture.
3. **`miss_table_<focus>.png`** + **`misses_<focus>.csv`** — the actual errors at the
   focus matcher's best-F1 threshold: **false merges** (e.g. Merck & Co ↔ Merck KGaA,
   HP ↔ HPE) and **missed matches** (acronyms/tickers/transliterations like
   Coca-Cola↔KO, トヨタ自動車↔Toyota).
4. **`embedding_map.png` / `.html`** — all variants in 2D (UMAP→t-SNE→PCA fallback),
   colored by cluster. Tight blobs = clean clusters; near-but-separate points are the
   confusables. The `.html` is hoverable.
5. **`explorer_<focus>.html`** — interactive strip plot of every pair's score, points
   colored as true-match / false-merge / missed-match / true-non-match; hover any
   point to read the two names.

Pick the focus matcher and map source:

```bash
python visualize.py --focus st_eridu --embed-matcher st_bge_m3 --reducer umap
```

---

## Suggested workflow

1. Run offline (`fuzzy_*`, `tfidf_char`) and read `metrics.csv` — note the AUC ceiling
   and that all hard negatives are merged at the F1-optimal point.
2. Install `sentence-transformers` and re-run. Compare `st_bge_m3` (generic) vs
   `st_lt_comp_en` / `st_eridu` (name-specialized) — the specialized models should lift
   AUC and shrink the hard-negative false-merge count. That reproduces the
   "specialized beats generic on short strings" result on data you control.
3. Use `score_distributions.png` to pick **two** thresholds (auto-accept / auto-reject)
   instead of one, and route the middle — the dual-threshold pattern.
4. Open `misses_<focus>.csv`, eyeball the residual errors, and decide which need an
   LLM adjudicator vs a human.

---

## Notes & caveats

- Ground-truth groupings reflect well-known public companies as of early 2026, with
  three deliberate judgment calls flagged in `make_dataset.py` (HP Inc. vs HPE kept
  **separate**; Google vs Alphabet **separate**; former names **merged**). Flip them in
  the `CLUSTERS`/`DROP` structures to match your domain.
- This is a *fast-iteration* set, intentionally small enough to read every error. For
  scale, graduate to LinkTransformer's bundled aliases, CompanyName2Vec, or
  OpenSanctions Pairs, and build a held-out set from your own data (GLEIF/LEI, SEC
  EDGAR former names, OpenCorporates).
- The best-F1 threshold is a defensible default but, on hard data, it's permissive.
  For an operating system you'd usually set a higher-precision auto-accept threshold
  and route the rest — exactly what the distributions plot helps you choose.
