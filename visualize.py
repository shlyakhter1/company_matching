#!/usr/bin/env python3
"""
visualize.py — turn benchmark output into pictures and miss reports.

Produces (in --outdir, default ./out):
  pr_curves.png            : precision-recall curve, all matchers overlaid
  score_distributions.png  : per-matcher score histograms, split pos / hard-neg / easy-neg
  misses_<focus>.csv       : every false-merge and missed-match at the focus matcher's best threshold
  miss_table_<focus>.png   : the worst false merges + missed matches, rendered
  embedding_map.png/.html  : 2D map of all name variants, colored by cluster (static + interactive)
  explorer_<focus>.html    : interactive strip plot of all pairs (hover = the two names)

Usage:
  python visualize.py                          # focus = best-F1 matcher, map from tfidf
  python visualize.py --focus tfidf_char
  python visualize.py --embed-matcher st_bge_m3 --reducer umap
"""
import os, csv, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm

# Register a CJK-capable fallback so transliterated names (トヨタ自動車, 中国银行, 삼성전자)
# render in the static PNGs instead of tofu boxes. Harmless if the font is absent.
for _fp in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/System/Library/Fonts/PingFang.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"]:
    if os.path.exists(_fp):
        try:
            _fm.fontManager.addfont(_fp)
        except Exception:
            pass
        break
plt.rcParams["font.family"] = ["DejaVu Sans", "Noto Sans CJK JP", "Noto Sans CJK SC",
                               "Noto Sans CJK KR", "WenQuanYi Zen Hei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

PAL = {"positive": "#1F8A5B", "hard_negative": "#DD6438", "easy_negative": "#9AB0BF"}


# ---------- loading ----------
def load(outdir):
    df = pd.read_csv(os.path.join(outdir, "pair_scores.csv"))
    met = pd.read_csv(os.path.join(outdir, "metrics.csv"))
    matcher_cols = [c for c in df.columns if c in set(met["matcher"])]
    return df, met, matcher_cols


def pr_curve(y, s, n=200):
    ths = np.linspace(s.min(), s.max(), n)
    P, R = [], []
    for t in ths:
        pred = s >= t
        tp = np.sum(pred & (y == 1)); fp = np.sum(pred & (y == 0)); fn = np.sum(~pred & (y == 1))
        P.append(tp / (tp + fp) if tp + fp else 1.0)
        R.append(tp / (tp + fn) if tp + fn else 0.0)
    return np.array(R), np.array(P)


# ---------- 1. PR curves ----------
def plot_pr(df, cols, outdir):
    y = df["label"].to_numpy()
    plt.figure(figsize=(7.5, 6))
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(cols)))
    for c, col in zip(cols, cmap):
        R, P = pr_curve(y, df[c].to_numpy())
        order = np.argsort(R)
        plt.plot(R[order], P[order], label=c, color=col, lw=2)
    base = (df["label"] == 1).mean()
    plt.axhline(base, ls="--", c="#888", lw=1, label=f"random ({base:.2f})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision–Recall by matcher")
    plt.xlim(0, 1.02); plt.ylim(0, 1.02)
    plt.legend(fontsize=8, loc="lower left"); plt.grid(alpha=.25)
    p = os.path.join(outdir, "pr_curves.png"); plt.tight_layout(); plt.savefig(p, dpi=140); plt.close()
    return p


# ---------- 2. score distributions ----------
def plot_distributions(df, cols, met, outdir):
    n = len(cols); ncol = min(3, n); nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 3.2 * nrow), squeeze=False)
    thr = dict(zip(met["matcher"], met["threshold"]))
    for k, c in enumerate(cols):
        ax = axes[k // ncol][k % ncol]
        for pt in ["positive", "hard_negative", "easy_negative"]:
            v = df.loc[df["pair_type"] == pt, c]
            ax.hist(v, bins=24, range=(0, 1), alpha=.6, color=PAL[pt], label=pt.replace("_", " "))
        if c in thr:
            ax.axvline(thr[c], color="#222", ls="--", lw=1.2)
            ax.text(thr[c], ax.get_ylim()[1]*.92, f" thr={thr[c]:.2f}", fontsize=8, va="top")
        ax.set_title(c, fontsize=11); ax.set_xlim(0, 1)
        if k == 0: ax.legend(fontsize=8)
    for k in range(n, nrow*ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle("Score distributions: positives should sit right, hard negatives are the overlap", y=1.0, fontsize=12)
    p = os.path.join(outdir, "score_distributions.png"); plt.tight_layout(); plt.savefig(p, dpi=140); plt.close()
    return p


# ---------- 3. miss table ----------
def miss_report(df, met, focus, outdir, top=18):
    thr = float(met.loc[met["matcher"] == focus, "threshold"].iloc[0])
    s = df[focus].to_numpy(); y = df["label"].to_numpy(); pred = s >= thr
    fp = df[(pred) & (y == 0)].copy(); fp["score"] = s[(pred) & (y == 0)]; fp["error"] = "false_merge"
    fn = df[(~pred) & (y == 1)].copy(); fn["score"] = s[(~pred) & (y == 1)]; fn["error"] = "missed_match"
    misses = pd.concat([fp.sort_values("score", ascending=False),
                        fn.sort_values("score")], ignore_index=True)
    keep = ["error", "name_a", "name_b", "pair_type", "cluster_a", "cluster_b", "score"]
    csv_path = os.path.join(outdir, f"misses_{focus}.csv")
    misses[keep].to_csv(csv_path, index=False)

    # render the worst offenders as a table image
    fm = fp.sort_values("score", ascending=False).head(top)
    mm = fn.sort_values("score").head(top)
    fig, axes = plt.subplots(1, 2, figsize=(15, 0.42 * max(len(fm), len(mm)) + 1.2))
    for ax, data, title, col in [
        (axes[0], fm, f"FALSE MERGES (predicted match, actually different)  ·  thr={thr:.2f}", "#DD6438"),
        (axes[1], mm, "MISSED MATCHES (same company, scored below threshold)", "#1F8A5B")]:
        ax.axis("off"); ax.set_title(title, fontsize=11, color=col, loc="left", pad=10)
        if len(data) == 0:
            ax.text(0, .5, "none 🎉", fontsize=12); continue
        tbl = ax.table(cellText=[[f"{r.name_a}", f"{r.name_b}", f"{r.score:.2f}"] for r in data.itertuples()],
                       colLabels=["name A", "name B", "score"], loc="center", cellLoc="left")
        tbl.auto_set_font_size(False); tbl.set_fontsize(8.5); tbl.scale(1, 1.35)
        for (row, _), cell in tbl.get_celld().items():
            cell.set_edgecolor("#DDD")
            if row == 0: cell.set_text_props(weight="bold"); cell.set_facecolor("#F2F6F8")
    png = os.path.join(outdir, f"miss_table_{focus}.png")
    plt.tight_layout(); plt.savefig(png, dpi=140, bbox_inches="tight"); plt.close()
    return csv_path, png, len(fp), len(fn), thr


# ---------- 4a. 2D embedding map ----------
def reduce_2d(X, reducer="umap", seed=42):
    reducer = reducer.lower()
    if reducer == "umap":
        try:
            import umap
            return umap.UMAP(n_neighbors=12, min_dist=0.1, metric="cosine",
                             random_state=seed).fit_transform(X), "UMAP"
        except Exception:
            reducer = "tsne"
    if reducer == "tsne":
        try:
            from sklearn.manifold import TSNE
            per = max(5, min(30, X.shape[0] // 4))
            return TSNE(n_components=2, perplexity=per, init="pca",
                        random_state=seed).fit_transform(X), "t-SNE"
        except Exception:
            reducer = "pca"
    from sklearn.decomposition import PCA
    return PCA(n_components=2, random_state=seed).fit_transform(X), "PCA"


def embedding_map(records_csv, embed_matcher, outdir, reducer="umap"):
    import matchers as M
    rec = pd.read_csv(records_csv)
    names = rec["name_variant"].astype(str).tolist()
    mt = next((m for m in M.default_registry() if m.name == embed_matcher), None)
    if mt is None:
        print(f"  [map] unknown matcher {embed_matcher}; using tfidf_char"); mt = M.TfidfMatcher()
    ok, why = M.available(mt)
    if not ok:
        print(f"  [map] {embed_matcher} unavailable ({why}); falling back to tfidf_char"); mt = M.TfidfMatcher()
    X = mt.embed(names)
    if X is None:
        print("  [map] matcher has no vector space; using tfidf_char"); X = M.TfidfMatcher().embed(names)
    XY, algo = reduce_2d(np.asarray(X), reducer)
    rec["x"], rec["y"] = XY[:, 0], XY[:, 1]

    # static png: color by cluster (repeating palette), draw faint within-cluster links
    # but only SHORT ones, so tight clusters show structure without a spaghetti web.
    fig, ax = plt.subplots(figsize=(11, 8))
    clusters = rec["cluster_id"].unique()
    cmap = plt.cm.tab20(np.linspace(0, 1, 20))
    cidx = {c: cmap[i % 20] for i, c in enumerate(clusters)}
    segs = []  # (x0,y0,x1,y1,color,length)
    for cid, g in rec.groupby("cluster_id"):
        if len(g) > 1:
            cx, cy = g["x"].mean(), g["y"].mean()
            for _, r in g.iterrows():
                d = float(np.hypot(r["x"] - cx, r["y"] - cy))
                segs.append((cx, cy, r["x"], r["y"], cidx[cid], d))
    if segs:
        cutoff = np.percentile([s[5] for s in segs], 55)  # drop the long crisscrossing links
        for cx, cy, x, y, col, d in segs:
            if d <= cutoff:
                ax.plot([cx, x], [cy, y], color=col, alpha=.30, lw=.7, zorder=1)
    ax.scatter(rec["x"], rec["y"], c=[cidx[c] for c in rec["cluster_id"]], s=24, zorder=2,
               edgecolors="white", linewidths=.4)
    ax.set_title(f"Name variants in 2D ({algo} of {mt.name}) — tight blobs = easy clusters, "
                 f"near-but-separate = confusables", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    png = os.path.join(outdir, "embedding_map.png"); plt.tight_layout(); plt.savefig(png, dpi=140); plt.close()

    # interactive html (hover shows the variant + cluster + canonical)
    try:
        import plotly.express as px
        fig = px.scatter(rec, x="x", y="y", color="cluster_id",
                         hover_data={"name_variant": True, "canonical_name": True,
                                     "variation_type": True, "x": False, "y": False, "cluster_id": False},
                         title=f"Name variants — {algo} of {mt.name} (hover to inspect)")
        fig.update_traces(marker=dict(size=8, line=dict(width=.4, color="white")))
        fig.update_layout(showlegend=False, height=720)
        html = os.path.join(outdir, "embedding_map.html")
        fig.write_html(html, include_plotlyjs="cdn")
    except Exception as e:
        html = None; print(f"  [map] interactive skipped: {e}")
    return png, html, algo, mt.name


# ---------- 4b. interactive explorer of pair scores ----------
def explorer(df, met, focus, outdir):
    try:
        import plotly.graph_objects as go
    except Exception as e:
        print(f"  [explorer] skipped: {e}"); return None
    thr = float(met.loc[met["matcher"] == focus, "threshold"].iloc[0])
    s = df[focus].to_numpy(); y = df["label"].to_numpy(); pred = s >= thr
    cat = np.where((pred) & (y == 1), "true match",
          np.where((pred) & (y == 0), "FALSE MERGE",
          np.where((~pred) & (y == 1), "MISSED MATCH", "true non-match")))
    colors = {"true match": "#1F8A5B", "FALSE MERGE": "#DD6438",
              "MISSED MATCH": "#C026D3", "true non-match": "#9AB0BF"}
    rng = np.random.default_rng(0)
    fig = go.Figure()
    for c in ["true non-match", "true match", "MISSED MATCH", "FALSE MERGE"]:
        m = cat == c
        fig.add_trace(go.Scatter(
            x=s[m], y=rng.uniform(0, 1, int(m.sum())), mode="markers",
            name=f"{c} ({int(m.sum())})",
            marker=dict(color=colors[c], size=7, opacity=.75, line=dict(width=.3, color="white")),
            text=[f"{a}  ⟷  {b}" for a, b in zip(df["name_a"][m], df["name_b"][m])],
            hovertemplate="%{text}<br>score=%{x:.3f}<extra></extra>"))
    fig.add_vline(x=thr, line_dash="dash", line_color="#222",
                  annotation_text=f"threshold {thr:.2f}", annotation_position="top")
    fig.update_layout(title=f"Pair scores — {focus} (hover any point to read the two names)",
                      xaxis_title=f"{focus} similarity", yaxis=dict(showticklabels=False, title="jitter"),
                      height=560, legend=dict(orientation="h", y=-0.18))
    html = os.path.join(outdir, f"explorer_{focus}.html")
    fig.write_html(html, include_plotlyjs="cdn")
    return html


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--records", default="data/company_records.csv")
    ap.add_argument("--focus", default=None, help="matcher name for misses/explorer (default: best F1)")
    ap.add_argument("--embed-matcher", default="tfidf_char", help="matcher used for the 2D map")
    ap.add_argument("--reducer", default="umap", choices=["umap", "tsne", "pca"])
    ap.add_argument("--no-map", action="store_true",
                    help="skip the 2D embedding map (for pair-only datasets with no records file)")
    args = ap.parse_args()

    df, met, cols = load(args.outdir)
    focus = args.focus or met.sort_values("f1", ascending=False)["matcher"].iloc[0]
    if focus not in cols:
        focus = cols[0]
    print(f"focus matcher: {focus}")

    print("  -> pr_curves.png");            plot_pr(df, cols, args.outdir)
    print("  -> score_distributions.png");  plot_distributions(df, cols, met, args.outdir)
    cpath, mpng, nfp, nfn, thr = miss_report(df, met, focus, args.outdir)
    print(f"  -> miss_table_{focus}.png  ({nfp} false merges, {nfn} missed @ thr={thr:.2f})")
    skip_map = "--no-map" if args.no_map else None
    if not skip_map and not os.path.exists(args.records):
        skip_map = f"records file {args.records} not found"
    if not skip_map:
        # A records file that doesn't cover these pairs (e.g. default records with an
        # OpenSanctions outdir) would render a map of the wrong dataset — detect and skip.
        recs = {r["name_variant"] for r in csv.DictReader(open(args.records, encoding="utf-8"))}
        pair_names = set(df["name_a"].astype(str)) | set(df["name_b"].astype(str))
        overlap = len(pair_names & recs) / max(len(pair_names), 1)
        if overlap < 0.05:
            skip_map = f"records file {args.records} doesn't match these pairs ({overlap:.0%} name overlap)"
    if skip_map:
        print(f"  -> embedding map skipped ({skip_map})")
    else:
        png, html, algo, used = embedding_map(args.records, args.embed_matcher, args.outdir, args.reducer)
        print(f"  -> embedding_map.png/.html  ({algo} of {used})")
    eh = explorer(df, met, focus, args.outdir)
    print(f"  -> explorer_{focus}.html")
    print("done.")


if __name__ == "__main__":
    main()
