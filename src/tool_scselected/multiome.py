import io

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import snapatac2 as snap


def scRNA_scATAC_perform_joint_embedding(adata_rna, adata_atac):
    """Joint spectral embedding of paired RNA + ATAC cells (SnapATAC2 multi_spectral).

    Uses the cells present in both objects, the highly variable genes of adata_rna and the
    selected features of adata_atac when those exist (all features otherwise). Returns a copy of
    the ATAC cells with obsm['X_joint'] and a joint UMAP; adata_rna and adata_atac are unchanged.
    """
    common = adata_rna.obs_names.intersection(adata_atac.obs_names)
    if len(common) == 0:
        raise ValueError("adata_rna and adata_atac share no cell names; the joint embedding needs "
                         "paired cells (the same barcodes in both objects).")
    print(f"Paired cells: {len(common)} (RNA {adata_rna.n_obs}, ATAC {adata_atac.n_obs})")
    rna = adata_rna[common]
    atac = adata_atac[common].copy()

    def _mask(a, col):
        return np.asarray(a.var[col], dtype=bool) if col in a.var else None

    def _sparse_view(a, mask, layers=()):
        """Minimal AnnData with a CSR matrix (multi_spectral needs sparse input). Uses the first
        of `layers` present (non-negative expression), else X."""
        import scipy.sparse as sp
        layer = next((l for l in layers if l in a.layers), None)
        x = a.layers[layer] if layer else a.X
        if mask is not None:
            x = x[:, mask]
        x = sp.csr_matrix(x, dtype=np.float64)
        x.data = np.clip(x.data, 0, None)   # spectral embedding needs non-negative values
        x.eliminate_zeros()
        return ad.AnnData(X=x, obs=pd.DataFrame(index=a.obs_names))

    rna_mask, atac_mask = _mask(rna, "highly_variable"), _mask(atac, "selected")
    print(f"Features: RNA {'highly variable genes' if rna_mask is not None else 'all genes'}, "
          f"ATAC {'selected features' if atac_mask is not None else 'all features'}")
    views = [_sparse_view(rna, rna_mask, layers=("log1p_norm", "counts")), _sparse_view(atac, atac_mask)]
    _, embedding = snap.tl.multi_spectral(views, features=None)
    atac.obsm["X_joint"] = embedding
    snap.tl.umap(atac, use_rep="X_joint", key_added="joint_umap", random_state=42)

    color = next((c for c in ("cell_type", "leiden") if c in atac.obs), None)
    if color is None:
        for c in ("cell_type", "leiden"):
            if c in rna.obs:
                atac.obs[c] = rna.obs[c].values
                color = c
                break
    fig = snap.pl.umap(atac, color=color, use_rep="X_joint_umap", show=False, interactive=False)
    fig.update_layout(title={"text": "scRNA_scATAC_joint_embedding", "x": 0.5, "xanchor": "center"})
    img = plt.imread(io.BytesIO(fig.to_image(format="png", scale=2)), format="png")
    f, ax = plt.subplots(figsize=(img.shape[1] / 200, img.shape[0] / 200))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title("scRNA_scATAC_joint_embedding")
    print("Joint embedding: obsm['X_joint']; joint UMAP: obsm['X_joint_umap']")
    print(atac)
    return atac

def scRNA_to_annotate_scATAC_clusters_using_scVI(adata_rna, adata_atac):
    """
    scRNA is annotated with cell types, scATAC is in the cell-by-bin matrix form.
    This is needed when scRNA and scATAC are not paired. Not needed for multiome data.
    """
    import scvi  # optional dependency (scvi-tools); only this unfinished helper needs it
    query = snap.pp.make_gene_matrix(adata_atac, gene_anno=snap.genome.hg38)
    query.obs['cell_type'] = pd.NA
    data = ad.concat([adata_rna, query],
        join='inner',
        label='batch',
        keys=["reference", "query"],
        index_unique='_',
    )
    sc.pp.filter_genes(data, min_cells=5)
    sc.pp.highly_variable_genes(
        data,
        n_top_genes = 5000,
        flavor="seurat_v3",
        batch_key="batch",
        subset=True
    )

    scvi.model.SCVI.setup_anndata(data, batch_key="batch")
    vae = scvi.model.SCVI(
        data,
        n_layers=2,
        n_latent=30,
        gene_likelihood="nb",
        dispersion="gene-batch",
    )

    vae.train(max_epochs=1000, early_stopping=True)
    ax = vae.history['elbo_train'][1:].plot()
    vae.history['elbo_validation'].plot(ax=ax)

    data.obs["celltype_scanvi"] = 'Unknown'
    ref_idx = data.obs['batch'] == "reference"
    data.obs["celltype_scanvi"][ref_idx] = data.obs['cell_type'][ref_idx]

    lvae = scvi.model.SCANVI.from_scvi_model(
        vae,
        adata=data,
        labels_key="celltype_scanvi",
        unlabeled_category="Unknown",
    )

    lvae.train(max_epochs=1000, n_samples_per_label=100)
    lvae.history['elbo_train'][1:].plot()

    # TODO: needs completion.











