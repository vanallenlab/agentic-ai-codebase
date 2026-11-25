import io
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import snapatac2 as snap

from src.tools_singlecell._type_coercion import (
    coerce_int, coerce_float, coerce_bool, coerce_str,
    coerce_str_list, coerce_int_list,
)

# The tools accept two kinds of scATAC objects:
#  - a fragment-level SnapATAC2 object (obsm['fragment_paired'] / ['fragment_single']), on which
#    every step can run from the raw fragments;
#  - a cell-by-peak count matrix (e.g. adata_atac in the HTAN tasks), which has no fragments.
#    Steps that need fragments (fragment-size / TSS QC, tile matrix, MACS3 peak calling) cannot
#    be recomputed on it; the other steps run on the peak counts.
_FRAGMENT_KEYS = ("fragment_paired", "fragment_single")
_NO_FRAGMENTS = (
    "adata_atac is a cell-by-peak count matrix without fragment data (no "
    "obsm['fragment_paired'] / obsm['fragment_single']), so {what} cannot be computed on it. "
    "The peak counts can be used directly: scATAC_feature_selection, "
    "scATAC_dimensionality_reduction, scATAC_clustering and scATAC_identify_marker_peaks "
    "work on this matrix."
)


def _has_fragments(adata):
    try:
        keys = list(adata.obsm.keys())
    except Exception:
        return False
    return any(k in keys for k in _FRAGMENT_KEYS)


def _selected_features(adata):
    """Boolean mask of the features chosen by scATAC_feature_selection, or None (all features)."""
    if "selected" in adata.var:
        return np.asarray(adata.var["selected"], dtype=bool)
    return None


def _show_plotly(fig, title):
    """Render a plotly figure into a matplotlib figure, so the harness captures it as a plot."""
    if fig is None:
        return
    fig.update_layout(title={"text": title, "x": 0.5, "xanchor": "center"})
    png = fig.to_image(format="png", scale=2)
    img = plt.imread(io.BytesIO(png), format="png")
    h, w = img.shape[:2]
    f, ax = plt.subplots(figsize=(w / 200, h / 200))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(title)
    f.tight_layout()


def _violins(adata, columns):
    """Violin + jitter plots of per-cell QC columns (column, title)."""
    columns = [(c, t) for c, t in columns if c in adata.obs.columns]
    if not columns:
        return
    fig, axes = plt.subplots(1, len(columns), figsize=(5 * len(columns), 4), squeeze=False)
    for ax, (col, title) in zip(axes[0], columns):
        data = pd.to_numeric(adata.obs[col], errors="coerce").dropna().values.astype(float)
        if len(data) == 0:
            continue
        parts = ax.violinplot(data, positions=[1], showmedians=False, showextrema=False, widths=0.8)
        for pc in parts["bodies"]:
            pc.set_facecolor("#D43F3A")
            pc.set_edgecolor("black")
            pc.set_linewidth(0.5)
            pc.set_alpha(0.6)
        jitter = np.random.normal(0, 0.12, size=len(data))
        ax.scatter(1 + jitter, data, c="black", s=0.5, alpha=0.2, rasterized=True, linewidths=0)
        ax.set_title(title, fontsize=11)
        ax.set_xlim(0.2, 1.8)
        ax.set_xticks([])
        ax.tick_params(axis="y", labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("scATAC_qc_metrics")
    fig.tight_layout()


def _first_column(adata, candidates, prefix=None):
    for c in candidates:
        if c in adata.obs.columns:
            return c
    if prefix:
        for c in adata.obs.columns:
            if c.startswith(prefix):
                return c
    return None


def scATAC_calculate_qc_metrics(adata_atac):
    """
    Calculate QC metrics for a scATAC-seq dataset.
    """
    if not _has_fragments(adata_atac):
        # Peak matrix: report and plot the per-cell QC that is already there.
        cols = [(_first_column(adata_atac, ["n_fragment"], prefix="nCount_"), "counts per cell"),
                (_first_column(adata_atac, ["tsse", "TSS.enrichment"]), "TSS enrichment"),
                (_first_column(adata_atac, ["nucleosome_signal"]), "nucleosome signal"),
                (_first_column(adata_atac, ["frac_dup", "blacklist_ratio"]), "duplicate / blacklist fraction")]
        cols = [(c, t) for c, t in cols if c]
        print(_NO_FRAGMENTS.format(what="fragment-size and TSS-enrichment QC"))
        if cols:
            print("Existing per-cell QC columns in adata_atac.obs (plotted):")
            print(adata_atac.obs[[c for c, _ in cols]].describe().T.to_string())
            _violins(adata_atac, cols)
        return adata_atac

    print("Calculating fragment size distribution...")
    snap.metrics.frag_size_distr(adata_atac)
    _show_plotly(snap.pl.frag_size_distr(adata_atac, show=False, interactive=False),
                 "scATAC_fragment_size_dist")

    if "fragment_paired" in adata_atac.obsm:
        print("Calculating nucleosome signal...")
        fragments = adata_atac.obsm["fragment_paired"]
        nucleosome_signal = []
        for i in range(fragments.shape[0]):
            sizes = fragments[i].data   # fragment sizes of this cell
            if len(sizes) == 0:
                nucleosome_signal.append(np.nan)
                continue
            nfr = np.sum(sizes < 147)
            mono = np.sum((sizes >= 147) & (sizes <= 294))
            nucleosome_signal.append(mono / nfr if nfr > 0 else np.nan)
        adata_atac.obs["nucleosome_signal"] = nucleosome_signal

    print("Calculating TSS enrichment...")
    snap.metrics.tsse(adata_atac, snap.genome.hg38)
    _show_plotly(snap.pl.tsse(adata_atac, interactive=False, show=False), "scATAC_tss_enrichment")

    _violins(adata_atac, [("n_fragment", "n_fragments"), ("tsse", "TSS_enrichment"),
                          ("nucleosome_signal", "nucleosome_signal")])
    print(adata_atac)
    return adata_atac


def scATAC_qc_filter_cells(adata_atac, min_counts=None, min_tsse=None, max_counts=None, max_nucleosome_signal=None):
    min_counts = coerce_int(min_counts)
    min_tsse = coerce_float(min_tsse)
    max_counts = coerce_int(max_counts)
    max_nucleosome_signal = coerce_float(max_nucleosome_signal)

    print("Filtering cells...")
    n_before = adata_atac.n_obs
    count_col = _first_column(adata_atac, ["n_fragment"], prefix="nCount_")
    tsse_col = _first_column(adata_atac, ["tsse", "TSS.enrichment"])
    if count_col is None and (min_counts is not None or max_counts is not None):
        counts = np.asarray(adata_atac.X.sum(axis=1)).ravel()
        count_label = "total peak counts (from X)"
    else:
        counts = adata_atac.obs[count_col].to_numpy(dtype=float) if count_col else None
        count_label = count_col

    keep = np.ones(adata_atac.n_obs, dtype=bool)
    if min_counts is not None:
        keep &= counts >= min_counts
    if max_counts is not None:
        keep &= counts <= max_counts
    if min_tsse is not None:
        if tsse_col is None:
            raise ValueError("No TSS enrichment column in adata_atac.obs; run scATAC_calculate_qc_metrics "
                             "on a fragment-level object first, or leave min_tsse unset.")
        keep &= adata_atac.obs[tsse_col].to_numpy(dtype=float) >= min_tsse
    if max_nucleosome_signal is not None:
        if "nucleosome_signal" not in adata_atac.obs:
            raise ValueError("No nucleosome_signal column in adata_atac.obs; leave max_nucleosome_signal unset.")
        keep &= adata_atac.obs["nucleosome_signal"].to_numpy(dtype=float) < max_nucleosome_signal

    used = [f"counts: {count_label}" if (min_counts is not None or max_counts is not None) else None,
            f"TSS: {tsse_col}" if min_tsse is not None else None,
            "nucleosome_signal" if max_nucleosome_signal is not None else None]
    print(f"Columns used: {', '.join(u for u in used if u) or 'none (no thresholds given)'}")
    if not keep.all():
        adata_atac = adata_atac[keep].copy()
    print(f"Kept {adata_atac.n_obs} of {n_before} cells")
    print(adata_atac)
    return adata_atac


def scATAC_calculate_tiles(adata_atac):
    if not _has_fragments(adata_atac):
        raise ValueError(_NO_FRAGMENTS.format(what="a tile matrix"))
    print("Calculating tiles...")
    if adata_atac.n_vars == 0:
        snap.pp.add_tile_matrix(adata_atac)          # adds n_vars (genomic bins)
        print(adata_atac)
        return adata_atac
    # The object already has a feature matrix: build the tile matrix as a new object and carry
    # over the cell annotations, fragments and metadata.
    tiles = snap.pp.add_tile_matrix(adata_atac, inplace=False)
    tiles.obs = adata_atac.obs.copy()
    for k in adata_atac.obsm.keys():
        tiles.obsm[k] = adata_atac.obsm[k]
    for k in adata_atac.uns.keys():
        tiles.uns[k] = adata_atac.uns[k]
    print(f"Replaced the existing {adata_atac.n_vars}-feature matrix with a {tiles.n_vars}-bin tile matrix")
    print(tiles)
    return tiles


def scATAC_feature_selection(adata_atac, n_features=250000):
    n_features = coerce_int(n_features, 250000)
    print("Selecting features...")
    n_features = min(n_features, adata_atac.n_vars)
    snap.pp.select_features(adata_atac, n_features=n_features, inplace=True)
    print(f"Selected {int(np.asarray(adata_atac.var['selected']).sum())} features")
    print(adata_atac)
    return adata_atac


def scATAC_doublet_detection(adata_atac):
    print("Detecting doublets...")
    features = _selected_features(adata_atac)
    if features is None:
        print("No feature selection found (adata_atac.var['selected']); using all features.")
    snap.pp.scrublet(adata_atac, features=features, inplace=True)
    fig, ax = plt.subplots()
    ax.hist(adata_atac.obs["doublet_probability"], bins=100)
    ax.set_xlabel("Doublet Probability")
    ax.set_ylabel("Number of Cells")
    ax.set_title("scATAC_doublet_detection")
    print(adata_atac.obs["doublet_probability"].describe().to_string())
    print(adata_atac)
    return adata_atac


def scATAC_filter_doublets(adata_atac, threshold=0.5):
    threshold = coerce_float(threshold, 0.5)
    if "doublet_probability" not in adata_atac.obs:
        raise ValueError("adata_atac.obs has no 'doublet_probability'; run scATAC_doublet_detection first.")
    print("Filtering doublets...")
    keep = adata_atac.obs["doublet_probability"].to_numpy(dtype=float) <= threshold
    n_before = adata_atac.n_obs
    if not keep.all():
        adata_atac = adata_atac[keep].copy()
    print(f"Removed {n_before - adata_atac.n_obs} of {n_before} cells (doublet_probability > {threshold})")
    print(adata_atac)
    return adata_atac


def scATAC_dimensionality_reduction(adata_atac, n_comps=100):
    n_comps = coerce_int(n_comps, 100)
    print("Performing dimensionality reduction using spectral embedding...")
    features = _selected_features(adata_atac)
    if features is None:
        print("No feature selection found (adata_atac.var['selected']); using all features.")
    snap.tl.spectral(adata_atac, n_comps=n_comps, features=features)

    # --- Eigenvalue elbow plot (analogous to PCA variance plot) ---
    eigenvalues = adata_atac.uns.get("spectral_eigenvalue", None)
    if eigenvalues is not None:
        eigenvalues = np.asarray(eigenvalues)
        n_display = min(len(eigenvalues), n_comps)
        components = np.arange(1, n_display + 1)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(components, eigenvalues[:n_display], "o-", markersize=3, linewidth=1)
        axes[0].set_xlabel("Component")
        axes[0].set_ylabel("Eigenvalue")
        axes[0].set_title("Spectral Eigenvalues (Elbow Plot)")
        axes[0].spines["top"].set_visible(False)
        axes[0].spines["right"].set_visible(False)

        total = eigenvalues[:n_display].sum()
        if total > 0:
            cum_prop = np.cumsum(eigenvalues[:n_display]) / total
            axes[1].plot(components, cum_prop, "o-", markersize=3, linewidth=1, color="tab:orange")
            axes[1].set_xlabel("Component")
            axes[1].set_ylabel("Cumulative Proportion")
            axes[1].set_title("Cumulative Eigenvalue Proportion")
            axes[1].axhline(0.9, ls="--", color="grey", lw=0.8, label="90%")
            axes[1].legend(fontsize=9)
            axes[1].spines["top"].set_visible(False)
            axes[1].spines["right"].set_visible(False)
        fig.tight_layout()
    else:
        print("Warning: spectral_eigenvalue not found in adata_atac.uns – skipping elbow plot.")

    print(adata_atac)
    return adata_atac


def scATAC_batch_correction(adata_atac, batch_key, method="harmony"):
    batch_key = coerce_str(batch_key)
    method = (coerce_str(method) or "harmony").lower()
    if "X_spectral" not in adata_atac.obsm:
        raise ValueError("adata_atac.obsm has no 'X_spectral'; run scATAC_dimensionality_reduction first.")
    if batch_key not in adata_atac.obs:
        raise ValueError(f"batch_key '{batch_key}' is not a column of adata_atac.obs "
                         f"(columns: {list(adata_atac.obs.columns)}).")
    print(f"Performing batch correction ({method}) on X_spectral by '{batch_key}'...")
    if method == "harmony":
        # harmonypy directly: SnapATAC2's wrapper assumes the pre-0.2 harmonypy output layout
        import harmonypy
        out = harmonypy.run_harmony(np.asarray(adata_atac.obsm["X_spectral"]), adata_atac.obs,
                                    [batch_key], random_state=42, max_iter_harmony=100)
        z = np.asarray(out.Z_corr)
        adata_atac.obsm["X_spectral_harmony"] = np.ascontiguousarray(z if z.shape[0] == adata_atac.n_obs else z.T, dtype=np.float64)
        key = "X_spectral_harmony"
    elif method == "scanorama":
        snap.pp.scanorama_integrate(adata_atac, batch=batch_key, use_rep="X_spectral")
        key = "X_spectral_scanorama"
    else:
        raise ValueError("Invalid method. Please choose either 'harmony' or 'scanorama'.")
    print(f"Corrected embedding stored in adata_atac.obsm['{key}'] (used by scATAC_clustering).")
    print(adata_atac)
    return adata_atac


def _clustering_rep(adata_atac):
    for key in ("X_spectral_harmony", "X_spectral_scanorama", "X_spectral"):
        if key in adata_atac.obsm:
            return key
    raise ValueError("adata_atac.obsm has no spectral embedding; run scATAC_dimensionality_reduction first.")


def scATAC_clustering(adata_atac, n_comps=2, n_neighbors=50, resolution=0.5):
    n_comps = coerce_int(n_comps, 2)
    n_neighbors = coerce_int(n_neighbors, 50)
    resolution = coerce_float(resolution, 0.5)
    use_rep = _clustering_rep(adata_atac)

    print(f"Performing KNN+Leiden clustering on {use_rep}...")
    snap.pp.knn(adata_atac, n_neighbors=n_neighbors, use_rep=use_rep)
    snap.tl.leiden(adata_atac, resolution=resolution)
    print("Performing UMAP...")
    snap.tl.umap(adata_atac, n_comps=n_comps, use_rep=use_rep, random_state=42)
    _show_plotly(snap.pl.umap(adata_atac, color="leiden", show=False, interactive=False),
                 "scATAC_umap_leiden")
    print(adata_atac.obs["leiden"].value_counts().sort_index().to_string())
    print(adata_atac)
    return adata_atac


def _region(name):
    """'chr1-100-600' or 'chr1:100-600' -> 'chr1:100-600' (SnapATAC2 region format)."""
    name = str(name)
    if ":" in name:
        return name
    chrom, start, end = name.rsplit("-", 2)
    return f"{chrom}:{start}-{end}"


def _candidate_peaks(peak_mat, in_g, rest, features, min_log_fc, max_candidates, min_pct=0.05):
    """Peaks open in >= min_pct of the group's cells and >= 2**min_log_fc times more often than
    in the rest; the max_candidates with the largest ratio."""
    names = peak_mat.var_names
    cols = np.arange(len(names)) if features is None else names.get_indexer(features)
    x = peak_mat.X
    frac_g = np.asarray((x[np.sort(in_g)] > 0).mean(axis=0)).ravel()[cols]
    frac_r = np.asarray((x[np.sort(rest)] > 0).mean(axis=0)).ravel()[cols]
    lfc = np.log2((frac_g + 1e-3) / (frac_r + 1e-3))
    ok = np.where((frac_g >= min_pct) & (lfc >= min_log_fc))[0]
    ok = ok[np.argsort(-lfc[ok])][:max_candidates]
    return list(names[cols[ok]])


def _marker_peaks(peak_mat, key, features=None, max_cells=2000, max_rest=5000, min_cells=10,
                  max_fdr=0.01, min_log_fc=1.0, top_n=1000, max_candidates=5000, seed=0):
    """Marker peaks per group: SnapATAC2 diff_test (logistic regression) of the group against
    all other cells, keeping peaks more accessible in the group (adjusted p < max_fdr,
    log2 fold change >= min_log_fc), best first, at most top_n per group. Cells are subsampled
    (max_cells per group, max_rest in the reference) to keep the test fast on large cohorts."""
    rng = np.random.default_rng(seed)
    labels = peak_mat.obs[key].astype(str).to_numpy()
    markers = {}
    for g in sorted(set(labels)):
        in_g = np.where(labels == g)[0]
        rest = np.where(labels != g)[0]
        if len(in_g) < min_cells or len(rest) < min_cells:
            print(f"  {g}: skipped ({len(in_g)} cells; need >= {min_cells} in the group and the rest)")
            continue
        if len(in_g) > max_cells:
            in_g = rng.choice(in_g, max_cells, replace=False)
        if len(rest) > max_rest:
            rest = rng.choice(rest, max_rest, replace=False)
        # Cheap pre-filter on the fraction of cells with the peak open, so the logistic-regression
        # test only runs on plausible candidates (at most max_candidates per group).
        cand = _candidate_peaks(peak_mat, in_g, rest, features, min_log_fc, max_candidates)
        if not cand:
            markers[g] = []
            continue
        res = snap.tl.diff_test(peak_mat, cell_group1=sorted(in_g.tolist()), cell_group2=sorted(rest.tolist()),
                                features=cand, direction="positive", min_log_fc=min_log_fc)
        res = res.to_pandas()
        res = res[res["adjusted p-value"] < max_fdr].sort_values(
            ["adjusted p-value", "log2(fold_change)"], ascending=[True, False])
        markers[g] = res["feature name"].head(top_n).tolist()
    return markers


# annotate cell types
def scATAC_identify_marker_peaks(adata_atac, cell_type_key, output_dir):
    cell_type_key = coerce_str(cell_type_key) or "leiden"
    output_dir = coerce_str(output_dir) or "."
    os.makedirs(output_dir, exist_ok=True)
    if cell_type_key not in adata_atac.obs:
        raise ValueError(f"cell_type_key '{cell_type_key}' is not a column of adata_atac.obs "
                         f"(columns: {list(adata_atac.obs.columns)}).")
    groups = adata_atac.obs[cell_type_key].astype(str)

    if _has_fragments(adata_atac):
        print(f"Calling peaks per '{cell_type_key}' group with MACS3...")
        group_peaks = snap.tl.macs3(adata_atac, groupby=cell_type_key, inplace=False)
        peaks = snap.tl.merge_peaks(group_peaks, snap.genome.hg38)
        peaks.write_csv(os.path.join(output_dir, "merged_peaks.csv"))
        print(f"{peaks.height} merged peaks -> merged_peaks.csv")
        peak_mat = snap.pp.make_peak_matrix(adata_atac, use_rep=peaks["Peaks"])
        peak_source_has_selection = None
    else:
        print("adata_atac has no fragment data: using its existing peaks (no MACS3 peak calling).")
        peak_mat = adata_atac
        peak_source_has_selection = _selected_features(adata_atac)
        if peak_source_has_selection is not None:
            print(f"Testing the {int(peak_source_has_selection.sum())} selected peaks (adata_atac.var['selected']).")
    # Light view for SnapATAC2: shares X, region names as chr:start-end, groups as plain strings
    # (unused categories would break the per-group aggregation).
    import anndata as ad
    peak_mat = ad.AnnData(X=peak_mat.X,
                          obs=pd.DataFrame({cell_type_key: groups.values}, index=peak_mat.obs_names),
                          var=pd.DataFrame(index=[_region(v) for v in peak_mat.var_names]))

    print("Finding marker peaks (each group vs. all other cells)...")
    features = None if peak_source_has_selection is None else list(peak_mat.var_names[peak_source_has_selection])
    markers = _marker_peaks(peak_mat, cell_type_key, features)
    rows = [(g, p) for g, ps in markers.items() for p in ps]
    pd.DataFrame(rows, columns=["group", "peak"]).to_csv(
        os.path.join(output_dir, "marker_peaks.csv"), index=False)
    counts = {g: len(ps) for g, ps in markers.items()}
    print("Marker peaks per group -> marker_peaks.csv:")
    print(pd.Series(counts).to_string())

    plot_markers = {g: list(ps)[:200] for g, ps in markers.items() if len(ps)}
    if plot_markers:
        _show_plotly(snap.pl.regions(peak_mat, groupby=cell_type_key, peaks=plot_markers,
                                     show=False, interactive=False),
                     "scATAC_marker_peaks")

    regions = {g: list(ps) for g, ps in markers.items() if len(ps) >= 5}
    if not regions:
        print("Too few marker peaks for motif enrichment (need >= 5 in a group).")
        return adata_atac
    print("Running motif enrichment (CIS-BP motifs, hg38)...")
    motifs = snap.tl.motif_enrichment(motifs=snap.datasets.cis_bp(unique=True), regions=regions,
                                      genome_fasta=snap.genome.hg38)
    table = pd.concat([df.to_pandas().assign(group=g) for g, df in motifs.items()], ignore_index=True)
    table.to_csv(os.path.join(output_dir, "motifs.csv"), index=False)
    print(f"Motif enrichment -> motifs.csv ({len(table)} rows). Top motifs per group:")
    fdr_col = next((c for c in table.columns if "adjusted" in c.lower() or "fdr" in c.lower()), None)
    fc_col = next((c for c in table.columns if "fold" in c.lower()), None)
    for g, sub in table.groupby("group"):
        if fdr_col and fc_col:
            sub = sub[sub[fc_col] > 0].sort_values([fdr_col, fc_col], ascending=[True, False])
        print(f"  {g}: " + (", ".join(sub.iloc[:8, 0].astype(str)) or "(no enriched motif)"))
    try:
        _show_plotly(snap.pl.motif_enrichment(motifs, max_fdr=0.0001, height=1200,
                                              interactive=False, show=False),
                     "scATAC_motif_enrichment")
    except Exception as e:  # e.g. no motif passes the FDR cut-off
        print(f"(motif enrichment plot skipped: {e})")
    return adata_atac


def scATAC_analyze_subset():
    # TODO: needs completion.
    pass
