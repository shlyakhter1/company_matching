#!/usr/bin/env python3
"""normalize.py — shared name-normalization utilities."""
import re, unicodedata

# legal forms across jurisdictions + a few connectives
LEGAL_SUFFIXES = {
 "inc","incorporated","corp","corporation","co","company","companies","ltd","limited",
 "llc","plc","sa","ag","se","nv","kk","kgaa","oao","pjsc","asa","lp","spa","srl","gmbh",
 "bv","oy","ab","as","aktiengesellschaft","holdings","holding","group","groupe",
}
CONNECTIVES = {"and","of","the","for","und","et","des","de","la","von","y"}

def fold_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def normalize(name: str, drop_legal=True, drop_connectives=True, fold=True) -> str:
    """Light, deterministic normalization suitable for blocking & lexical scoring."""
    s = name.lower().strip()
    s = s.replace("&", " and ")
    if fold:
        s = fold_accents(s)
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)   # drop punctuation
    s = re.sub(r"\s+", " ", s).strip()
    toks = s.split()
    if drop_legal:
        toks = [t for t in toks if t not in LEGAL_SUFFIXES]
    if drop_connectives:
        toks = [t for t in toks if t not in CONNECTIVES]
    return " ".join(toks) if toks else s   # never return empty

def tokens(name: str) -> set:
    return set(normalize(name).split())

if __name__ == "__main__":
    for n in ["Apple Computer, Inc.", "Nestlé S.A.", "Procter & Gamble Co.",
              "AT&T Inc.", "Bayerische Motoren Werke AG", "Газпром"]:
        print(f"{n!r:45} -> {normalize(n)!r}")
