# Hybrid router comparison (router + LLM band, fixed band per stats)

| pipeline | router | band | reused | LLM band acc | P | R | F1 | misses | FP | best-F1 sweep | ROC-AUC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hybrid_tfidf_llm | tfidf | 438 | 438 | 1.000 | 0.987 | 0.647 | 0.782 | 165 | 4 | 0.885 @ 0.02 | 0.881 |
| hybrid_alias_llm | alias | 488 | 488 | 1.000 | 0.989 | 0.754 | 0.856 | 115 | 4 | 0.908 @ 0.03 | 0.903 |

Band = [0.1, 0.95]; composite: in-band -> 0.98/0.02 by LLM verdict, outside -> router score.
