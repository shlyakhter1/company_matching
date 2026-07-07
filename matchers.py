#!/usr/bin/env python3
"""
matchers.py — pluggable name matchers behind one interface.

Every matcher implements:
  .name                      -> str (column label in results)
  .needs                     -> list of pip deps (for a friendly skip message)
  .score_pairs(pairs)        -> np.ndarray of similarity in [0,1], one per (a,b)
  .embed(names) (optional)   -> np.ndarray [n, d] vectors, for the UMAP map; or None

Backends:
  FuzzyMatcher              - RapidFuzz, several scorers, optional normalization (offline)
  TfidfMatcher              - char n-gram TF-IDF cosine (offline, no downloads)
  SentenceTransformerMatcher- bge-m3 / lt-wikidata-comp / eridu  (downloads from HF)
  OpenAIMatcher             - text-embedding-3-* via API key            (network)
  CohereMatcher             - embed-v4 via API key                      (network)

Run `python matchers.py` to see which backends are available in this environment.
"""
from __future__ import annotations
import os, numpy as np

def _cos(a, b):
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return np.sum(a * b, axis=1)

# ----------------------------------------------------------------------------
class FuzzyMatcher:
    def __init__(self, scorer="token_sort", normalize=True):
        from rapidfuzz import fuzz
        self._f = {"token_sort": fuzz.token_sort_ratio,
                   "token_set":  fuzz.token_set_ratio,
                   "wratio":     fuzz.WRatio,
                   "ratio":      fuzz.ratio}[scorer]
        self.scorer = scorer
        self.do_norm = normalize
        self.name = f"fuzzy_{scorer}" + ("_norm" if normalize else "")
        self.needs = ["rapidfuzz"]

    def _prep(self, s):
        if self.do_norm:
            from normalize import normalize as nz
            return nz(s)
        return s

    def score_pairs(self, pairs):
        return np.array([self._f(self._prep(a), self._prep(b)) / 100.0 for a, b in pairs])

    def embed(self, names):
        return None   # fuzzy has no vector space

# ----------------------------------------------------------------------------
class TfidfMatcher:
    """Character n-gram TF-IDF — a strong, fully-offline name-matching baseline."""
    def __init__(self, analyzer="char_wb", ngram=(2, 4), normalize=True):
        self.analyzer, self.ngram, self.do_norm = analyzer, ngram, normalize
        self.name = "tfidf_char"
        self.needs = ["scikit-learn"]
        self._vec = None

    def _prep(self, names):
        if self.do_norm:
            from normalize import normalize as nz
            return [nz(s) for s in names]
        return list(names)

    def _fit_transform(self, names):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._vec = TfidfVectorizer(analyzer=self.analyzer, ngram_range=self.ngram)
        X = self._vec.fit_transform(self._prep(names))
        return X

    def score_pairs(self, pairs):
        names = [x for p in pairs for x in p]
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(analyzer=self.analyzer, ngram_range=self.ngram)
        X = vec.fit_transform(self._prep(names))
        A, B = X[0::2], X[1::2]
        num = np.asarray(A.multiply(B).sum(axis=1)).ravel()
        den = (np.sqrt(np.asarray(A.multiply(A).sum(1)).ravel()) *
               np.sqrt(np.asarray(B.multiply(B).sum(1)).ravel()) + 1e-9)
        return num / den

    def embed(self, names):
        X = self._fit_transform(names)
        return np.asarray(X.todense())

# ----------------------------------------------------------------------------
class SentenceTransformerMatcher:
    """Local HF models — e.g. BAAI/bge-m3, dell-research-harvard/lt-wikidata-comp-en, Graphlet-AI/eridu."""
    def __init__(self, model_id, short=None, normalize=False):
        self.model_id = model_id
        self.do_norm = normalize
        self.name = "st_" + (short or model_id.split("/")[-1].replace("-", "_"))
        self.needs = ["sentence-transformers", "torch"]
        self._m = None

    def _model(self):
        if self._m is None:
            from sentence_transformers import SentenceTransformer
            self._m = SentenceTransformer(self.model_id)
        return self._m

    def _prep(self, names):
        if self.do_norm:
            from normalize import normalize as nz
            return [nz(s) for s in names]
        return list(names)

    def embed(self, names):
        emb = self._model().encode(self._prep(names), convert_to_numpy=True,
                                   normalize_embeddings=True, show_progress_bar=False)
        return emb

    def score_pairs(self, pairs):
        a = self.embed([p[0] for p in pairs])
        b = self.embed([p[1] for p in pairs])
        return (_cos(a, b) + 1) / 2   # map [-1,1] -> [0,1]

# ----------------------------------------------------------------------------
class OpenAIMatcher:
    def __init__(self, model="text-embedding-3-small"):
        self.model = model
        self.name = "openai_" + model.split("-")[-1]
        self.needs = ["openai (+ OPENAI_API_KEY)"]
        self._c = None

    def _client(self):
        if self._c is None:
            from openai import OpenAI
            self._c = OpenAI()   # reads OPENAI_API_KEY
        return self._c

    def embed(self, names, batch=256):
        out = []
        for i in range(0, len(names), batch):
            chunk = [s if s.strip() else " " for s in names[i:i+batch]]
            r = self._client().embeddings.create(model=self.model, input=chunk)
            out.extend([d.embedding for d in r.data])
        return np.array(out)

    def score_pairs(self, pairs):
        a = self.embed([p[0] for p in pairs]); b = self.embed([p[1] for p in pairs])
        return (_cos(a, b) + 1) / 2

# ----------------------------------------------------------------------------
class CohereMatcher:
    def __init__(self, model="embed-v4.0"):
        self.model = model
        self.name = "cohere_v4"
        self.needs = ["cohere (+ COHERE_API_KEY)"]
        self._c = None

    def _client(self):
        if self._c is None:
            import cohere
            self._c = cohere.ClientV2(os.environ["COHERE_API_KEY"])
        return self._c

    def embed(self, names, batch=96, input_type="search_document"):
        out = []
        for i in range(0, len(names), batch):
            chunk = [s if s.strip() else " " for s in names[i:i+batch]]
            r = self._client().embed(texts=chunk, model=self.model,
                                     input_type=input_type, embedding_types=["float"])
            out.extend(r.embeddings.float)
        return np.array(out)

    def score_pairs(self, pairs):
        a = self.embed([p[0] for p in pairs]); b = self.embed([p[1] for p in pairs])
        return (_cos(a, b) + 1) / 2

# ----------------------------------------------------------------------------
class AliasRouter:
    """TF-IDF augmented with classic alias rules (initialisms + ticker-style
    contractions). A fully-offline step toward an embedding router: rescues
    zero-character-overlap variants like 'AMZN'<->'Amazon.com, Inc.' or
    'GS'<->'Goldman Sachs' into the LLM band instead of auto-rejecting them.
    Rules score 0.85-0.92, deliberately below the auto-accept threshold, so
    rescued pairs are routed to the adjudicator rather than trusted blindly."""
    def __init__(self):
        self._tfidf = TfidfMatcher()
        self.name = "alias_router"
        self.needs = ["scikit-learn"]

    @staticmethod
    def _initials(name):
        from normalize import normalize
        toks = normalize(name).split()
        return "".join(t[0] for t in toks) if len(toks) >= 2 else ""

    @staticmethod
    def _is_contraction(short, long_name):
        from normalize import normalize
        s = normalize(short).replace(" ", "")
        if not (3 <= len(s) <= 6):
            return False
        for t in normalize(long_name).split():
            if len(t) > len(s) and t[0] == s[0]:
                it = iter(t)
                if all(c in it for c in s):
                    return True
        return False

    def _rule(self, a, b):
        from normalize import normalize
        na, nb = normalize(a), normalize(b)
        sc = 0.0
        for s, l in ((na, b), (nb, a)):
            if len(s.split()) == 1 and 2 <= len(s) <= 6 and s == self._initials(l):
                sc = max(sc, 0.92)
        if len(na.split()) == 1 and self._is_contraction(a, b): sc = max(sc, 0.85)
        if len(nb.split()) == 1 and self._is_contraction(b, a): sc = max(sc, 0.85)
        return sc

    def score_pairs(self, pairs):
        base = self._tfidf.score_pairs(pairs)
        return np.maximum(base, np.array([self._rule(a, b) for a, b in pairs]))

    def embed(self, names):
        return self._tfidf.embed(names)

# ----------------------------------------------------------------------------
def default_registry():
    """The matchers run by benchmark.py unless --matchers is given.
    Embedding/API backends are included but skipped gracefully if unavailable."""
    return [
        FuzzyMatcher("token_sort", normalize=False),
        FuzzyMatcher("token_sort", normalize=True),
        FuzzyMatcher("token_set",  normalize=True),
        TfidfMatcher(),
        AliasRouter(),
        SentenceTransformerMatcher("BAAI/bge-m3", short="bge_m3"),
        SentenceTransformerMatcher("dell-research-harvard/lt-wikidata-comp-en", short="lt_comp_en"),
        SentenceTransformerMatcher("Graphlet-AI/eridu", short="eridu"),
        OpenAIMatcher("text-embedding-3-small"),
        CohereMatcher("embed-v4.0"),
    ]

def available(m):
    """Return (ok, reason). Tries a 1-pair smoke test without heavy downloads where possible."""
    try:
        if isinstance(m, FuzzyMatcher):
            import rapidfuzz; return True, ""
        if isinstance(m, TfidfMatcher):
            import sklearn; return True, ""
        if isinstance(m, SentenceTransformerMatcher):
            import sentence_transformers; return True, "(will download model on first use)"
        if isinstance(m, OpenAIMatcher):
            import openai
            return (("OPENAI_API_KEY" in os.environ), "set OPENAI_API_KEY")
        if isinstance(m, CohereMatcher):
            import cohere
            return (("COHERE_API_KEY" in os.environ), "set COHERE_API_KEY")
    except Exception as e:
        return False, f"missing dep: {e.__class__.__name__}"
    return True, ""

if __name__ == "__main__":
    print("Matcher availability in this environment:")
    for m in default_registry():
        ok, why = available(m)
        print(f"  {'OK ' if ok else 'SKIP'}  {m.name:22} {why}")
