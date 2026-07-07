# Hybrid router comparison (router + LLM band, fixed band per stats)

| pipeline | router | band | reused | LLM band acc | P | R | F1 | misses | FP | best-F1 sweep | ROC-AUC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hybrid_tfidf_llm | tfidf | 438 | 438 | 1.000 | 0.987 | 0.647 | 0.782 | 165 | 4 | 0.885 @ 0.02 | 0.881 |
| hybrid_alias_llm | alias | 488 | 488 | 1.000 | 0.989 | 0.754 | 0.856 | 115 | 4 | 0.908 @ 0.03 | 0.903 |
| hybrid_ltcomp_llm | lt_comp_en | 596 | 596 | 0.993 | 0.996 | 0.962 | 0.978 | 18 | 2 | 0.978 @ 0.95 | 0.976 |
| hybrid_ltmulti_llm | lt_comp_multi | 697 | 697 | 0.994 | 0.998 | 0.983 | 0.990 | 8 | 1 | 0.990 @ 0.95 | 0.990 |

Band(s) = [0.1, 0.95]; composite: in-band -> 0.98/0.02 by LLM verdict, outside -> router score.
