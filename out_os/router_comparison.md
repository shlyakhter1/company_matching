# Hybrid router comparison (router + LLM band, fixed band per stats)

| pipeline | router | band | reused | LLM band acc | P | R | F1 | misses | FP | best-F1 sweep | ROC-AUC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hybrid_tfidf_llm | tfidf | 1965 | 1965 | 0.898 | 0.838 | 0.662 | 0.740 | 844 | 321 | 0.740 @ 0.10 | 0.695 |
| hybrid_alias_llm | alias | 1968 | 1968 | 0.897 | 0.837 | 0.663 | 0.740 | 843 | 322 | 0.740 @ 0.10 | 0.695 |
| hybrid_ltcomp_llm | lt_comp_en | 3085 | 3085 | 0.896 | 0.835 | 0.873 | 0.854 | 317 | 431 | 0.855 @ 0.09 | 0.852 |
| hybrid_ltmulti_llm | lt_comp_multi | 3764 | 3764 | 0.885 | 0.843 | 0.904 | 0.872 | 241 | 421 | 0.873 @ 0.10 | 0.868 |

Band = [0.1, 0.95]; composite: in-band -> 0.98/0.02 by LLM verdict, outside -> router score.
