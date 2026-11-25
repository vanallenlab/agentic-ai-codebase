"""
Tool registry for scselected scATAC-seq tools.

Curated subset of single-cell ATAC tools using SnapATAC2.
Category: scselected_scatac
Implementation: src.tool_scselected.scatac
"""

description = [
    {
        "name": "scATAC_calculate_qc_metrics",
        "description": "Calculate QC metrics for scATAC-seq data using SnapATAC2. On a fragment-level object, computes fragment size distribution, nucleosome signal, and TSS enrichment (hg38) and returns violin plots of n_fragments, TSS enrichment, and nucleosome signal. On a cell-by-peak matrix without fragments (no obsm['fragment_paired']), it summarizes and plots the per-cell QC columns already in adata.obs instead.",
        "required_parameters": [],
        "optional_parameters": [],
        "is_plotting": True,
    },
    {
        "name": "scATAC_qc_filter_cells",
        "description": "Filter cells based on QC metrics: min_counts / max_counts (n_fragment, an nCount_* column, or total counts in X), min_tsse (tsse or TSS.enrichment column) and max_nucleosome_signal. Should be called after scATAC_calculate_qc_metrics.",
        "required_parameters": [],
        "optional_parameters": [
            {
                "name": "min_counts",
                "type": "int",
                "description": "Minimum total counts per cell.",
                "default": None
            },
            {
                "name": "min_tsse",
                "type": "float",
                "description": "Minimum TSS enrichment score per cell.",
                "default": None
            },
            {
                "name": "max_counts",
                "type": "int",
                "description": "Maximum total counts per cell.",
                "default": None
            },
            {
                "name": "max_nucleosome_signal",
                "type": "float",
                "description": "Maximum nucleosome signal. Cells above are removed (poor fragmentation).",
                "default": None
            },
        ],
    },
    {
        "name": "scATAC_calculate_tiles",
        "description": "Add a tile matrix (genomic bins) using snap.pp.add_tile_matrix. Requires fragment data (obsm['fragment_paired']); replaces any existing feature matrix. Not needed for a cell-by-peak matrix, whose peaks can be used directly.",
        "required_parameters": [],
        "optional_parameters": [],
    },
    {
        "name": "scATAC_feature_selection",
        "description": "Select top features (genomic bins or peaks) using snap.pp.select_features; marks them in adata.var['selected'], which the doublet detection and spectral embedding then use.",
        "required_parameters": [],
        "optional_parameters": [
            {
                "name": "n_features",
                "type": "int",
                "description": "Number of top features to select.",
                "default": 250000
            },
        ],
    },
    {
        "name": "scATAC_doublet_detection",
        "description": "Detect doublets using Scrublet via snap.pp.scrublet. Adds doublet_probability to adata.obs and generates a histogram of doublet probabilities. Does NOT remove doublets.",
        "required_parameters": [],
        "optional_parameters": [],
        "is_plotting": True,
    },
    {
        "name": "scATAC_filter_doublets",
        "description": "Remove detected doublets based on a probability threshold. Should be called after scATAC_doublet_detection.",
        "required_parameters": [],
        "optional_parameters": [
            {
                "name": "threshold",
                "type": "float",
                "description": "Doublet probability threshold. Cells above are removed.",
                "default": 0.5
            },
        ],
    },
    {
        "name": "scATAC_dimensionality_reduction",
        "description": "Perform spectral embedding dimensionality reduction using snap.tl.spectral. Returns eigenvalue elbow plot and cumulative proportion plot.",
        "required_parameters": [],
        "optional_parameters": [
            {
                "name": "n_comps",
                "type": "int",
                "description": "Number of spectral components to compute.",
                "default": 100
            },
        ],
        "is_plotting": True,
    },
    {
        "name": "scATAC_batch_correction",
        "description": "Perform batch correction of the spectral embedding using Harmony or Scanorama via SnapATAC2 (stored as obsm['X_spectral_harmony'] or obsm['X_spectral_scanorama'], which scATAC_clustering then uses). Requires scATAC_dimensionality_reduction first.",
        "required_parameters": [
            {
                "name": "batch_key",
                "type": "str",
                "description": "Column in adata.obs identifying batches."
            },
        ],
        "optional_parameters": [
            {
                "name": "method",
                "type": "str",
                "description": "Batch correction method: 'harmony' or 'scanorama'.",
                "default": "harmony"
            },
        ],
    },
    {
        "name": "scATAC_clustering",
        "description": "Perform KNN graph construction, Leiden clustering, and UMAP on scATAC data using the batch-corrected spectral embedding if present, else X_spectral. Returns UMAP plot colored by leiden cluster.",
        "required_parameters": [],
        "optional_parameters": [
            {
                "name": "n_comps",
                "type": "int",
                "description": "Number of UMAP components.",
                "default": 2
            },
            {
                "name": "n_neighbors",
                "type": "int",
                "description": "Number of neighbors for KNN graph.",
                "default": 50
            },
            {
                "name": "resolution",
                "type": "float",
                "description": "Resolution parameter for Leiden clustering (higher = more clusters).",
                "default": 0.5
            },
        ],
        "is_plotting": True,
    },
    {
        "name": "scATAC_identify_marker_peaks",
        "description": "Identify marker peaks per group and their enriched motifs via SnapATAC2 (CIS-BP motifs, hg38). On a fragment-level object, first calls peaks per group with MACS3 and builds a peak matrix; on a cell-by-peak matrix, uses its existing peaks. Saves marker_peaks.csv, motifs.csv (and merged_peaks.csv when peaks are called) to output_dir and prints the top motifs per group.",
        "required_parameters": [
            {
                "name": "cell_type_key",
                "type": "str",
                "description": "Column in adata.obs containing cell type annotations for grouping."
            },
        ],
        "optional_parameters": [
            {
                "name": "output_dir",
                "type": "str",
                "description": "Directory to save output CSV files (marker_peaks.csv, motifs.csv, merged_peaks.csv).",
                "default": "."
            },
        ],
        "is_plotting": True,
    },
]
