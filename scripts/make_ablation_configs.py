#!/usr/bin/env python3
"""Generate modality x image ablation task prompts and configs for the HTAN benchmark.

Arms (modality x vision):
    multiome_vision  scRNA + scATAC, plots shown to the model (current setup)
    multiome_text    scRNA + scATAC, plots saved but never shown
    rna_vision / rna_text      scRNA only
    atac_vision / atac_text    scATAC only

All arms: capture_telemetry false (no per-step post-hoc reasoning call),
prompt caching on. cNMF tables (RNA-derived) are given to every arm, since the
programs are the object being ranked.

Task prompts are produced by rule-based edits of the original task files:
  * all arms: "Support claims with plots or statistics." ->
    "Generate plots and statistics to support claims." (identical in vision and
    text arms, so a vision/text pair differs only by the access notice below)
  * text arms: followed by "You do not have access to the plots you generate:
    they are saved to disk, but images are never shown to you in this run."
  * rna/atac arms: data description, routing, the "use both ... data" sentence
    and the technical notes are rewritten for the single modality; ATAC-only
    cell-type annotation is over the cells of `adata_atac`.
Analysis goals and deliverables are otherwise verbatim. Every variant is
checked for leftover references to the dropped modality.

Usage:
    python scripts/make_ablation_configs.py                         # all cancers, claude-opus-5-5
    python scripts/make_ablation_configs.py --cancers brca          # pilot
    python scripts/make_ablation_configs.py --model claude-opus-4-6 # another model
    python scripts/make_ablation_configs.py --check                 # dry run, no writes

Configs go to config/htan_ablation/<model>/<arm>/<cancer>/<task>.yaml, run
outputs to output/htan_ablation/<model>/<arm>/<cancer>/<task>/. Task prompts
(tasks/htan_ablation/<arm>/...) do not depend on the model.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
SRC_CONFIGS = REPO / "config" / "htan" / "opus-4-6"
OUT_CONFIGS = REPO / "config" / "htan_ablation"
OUT_TASKS = REPO / "tasks" / "htan_ablation"
WANDB_PROJECT = "agentic-ai-htan-ablation"

DEFAULT_MODEL = "claude-opus-5-5"

# Per-model llm settings on top of provider/model/prompt_caching. Models with
# always-on thinking: room for thinking + reply, explicit effort (never changed
# mid-run: that would invalidate the prompt cache), readable thinking summaries
# for the step telemetry (model_thinking).
MODEL_LLM = {
    "claude-opus-5-5": {"max_tokens": 64000, "effort": "medium", "thinking_display": "summarized"},
    "claude-opus-4-6": {},
}

MODALITIES = ("multiome", "rna", "atac")
VISION = {"vision": True, "text": False}
TOOLS = {
    "multiome": None,  # keep the original tool_category
    "rna": "scselected_scrna, external_database",
    "atac": "scselected_scatac, external_database",
}

HEADER_ORIG = ("You are a computational biology expert analyzing single-cell multiome "
               "(scRNA-seq + scATAC-seq) data from tumor samples.")
HEADER = {
    "rna": "You are a computational biology expert analyzing single-cell RNA-seq (scRNA-seq) data from tumor samples.",
    "atac": "You are a computational biology expert analyzing single-cell ATAC-seq (scATAC-seq) data from tumor samples.",
}
TECH_ORIG = "in addition to scRNA, scATAC and scMultiome tools."
TECH = {"rna": "in addition to scRNA tools.", "atac": "in addition to scATAC tools."}
PAIRING = (" scATAC cells maybe be strict subset of scRNA cells (same barcodes/labels where present) "
           "when paired sequencing was performed, else different barcodes.")
NUMBER = {1: "One data object is", 2: "Two data objects are", 3: "Three data objects are",
          4: "Four data objects are", 5: "Five data objects are", 6: "Six data objects are"}
CNMF_ROUTING = (" The cNMF tables (`usage_df`, `spectra_scores_df`, `top_genes_df`) are passed to the tools "
                "that take them, and are available in `run_python_repl` by the same names.")


PLOTS_RX = r"Support claims with (single-panel )?plots or statistics\."
NO_PLOT_ACCESS = (" You do not have access to the plots you generate: they are saved to disk, "
                  "but images are never shown to you in this run.")


class RewriteError(Exception):
    pass


def _sub(text: str, pattern: str, repl: str, *, count_expected: int | None = 1, flags=0) -> str:
    new, n = re.subn(pattern, repl, text, flags=flags)
    if count_expected is not None and n != count_expected:
        raise RewriteError(f"pattern matched {n}x (expected {count_expected}): {pattern[:80]}")
    return new


def rewrite_plots(text: str, vision: bool) -> str:
    """Same plot instruction in every arm; text arms add the no-access notice."""
    return _sub(text, PLOTS_RX, lambda m: f"Generate {m.group(1) or ''}plots and statistics to support claims."
                + ("" if vision else NO_PLOT_ACCESS))


def rewrite(text: str, modality: str) -> str:
    """Single-modality variant of a multiome task prompt."""
    if HEADER_ORIG not in text or TECH_ORIG not in text:
        raise RewriteError("task does not follow the multiome template")
    has_backed = "`adata_atac_backed`" in text
    t = text.replace(HEADER_ORIG, HEADER[modality]).replace(TECH_ORIG, TECH[modality])

    rna_bullet = r"^- \*\*`adata_rna`\*\* — .*\n"
    atac_bullet = r"^- \*\*`adata_atac`\*\* — .*\n"
    backed_bullet = r"^- \*\*`adata_atac_backed`\*\* — .*\n"
    if modality == "rna":
        t = _sub(t, atac_bullet, "", flags=re.M)
        if has_backed:
            t = _sub(t, backed_bullet, "", flags=re.M)
        n_objects = 1
        routing = "**Data routing:** scRNA_ tools receive `adata_rna`. In `run_python_repl`, use `adata_rna` by name."
        t = _sub(t, r"Use both scRNA and scATAC data to", "Use the scRNA data to")
        t = _sub(t, r"from marker genes, chromatin accessibility, or pathway activity\.",
                 "from marker genes or pathway activity.", count_expected=None)
    else:
        t = _sub(t, rna_bullet, "", flags=re.M)
        t = t.replace(PAIRING, "")
        n_objects = 2 if has_backed else 1
        if has_backed:
            routing = ("**Data routing:** scATAC_ tools receive `adata_atac` (or `adata_atac_backed` when the "
                       "tool accepts it). In `run_python_repl`, use `adata_atac` and/or `adata_atac_backed` by name.")
        else:
            routing = "**Data routing:** scATAC_ tools receive `adata_atac`. In `run_python_repl`, use `adata_atac` by name."
        t = _sub(t, r"Use both scRNA and scATAC data to", "Use the scATAC data to")
        # Cell-type annotation is over the ATAC cells (no adata_rna is loaded)
        t = _sub(t, r"`adata_rna`", "`adata_atac`", count_expected=None)
        # cNMF usage is per scRNA cell: say how its barcodes relate to adata_atac
        t = _sub(t, r"(- \*\*`usage_df`\*\* — scores per cell for all cNMF program \(only for the tumor compartment\)\.)",
                 r"\1 Cell barcodes come from the scRNA assay; they match `adata_atac` barcodes where paired "
                 r"sequencing was performed.", count_expected=None)

    # Count every object still listed (AnnData objects and cNMF tables alike)
    n_objects = len(re.findall(r"^- \*\*`[A-Za-z_]+`\*\* — ", t.split("### Analysis Goals")[0], flags=re.M))
    if "`usage_df`" in t:
        routing += CNMF_ROUTING
    t = _sub(t, r"^\w+ data objects? (?:are|is) pre-loaded", NUMBER[n_objects] + " pre-loaded", flags=re.M)
    t = _sub(t, r"^\*\*Data routing:\*\*.*$", routing, flags=re.M)
    _check_leftovers(t, modality)
    return t


def _check_leftovers(t: str, modality: str):
    if modality == "rna":
        bad = re.findall(r"[^\n]*(?:ATAC|atac|chromatin|accessib|peak|motif|fragment)[^\n]*", t)
    else:
        bad = re.findall(r"[^\n]*(?:adata_rna|scRNA_ tools|expression matri|Use both)[^\n]*", t)
    if bad:
        raise RewriteError(f"{modality} variant still mentions the dropped modality:\n  " + "\n  ".join(bad))


def make_config(src_cfg: dict, cancer: str, task: str, modality: str, vision: str, task_file: Path,
                model: str) -> dict:
    cfg = dict(src_cfg)
    arm = f"{modality}_{vision}"
    short = model.removeprefix("claude-")
    cfg["task_file"] = str(task_file.relative_to(REPO))
    if modality == "rna":
        cfg.pop("scatac_data_path", None)
        cfg.pop("scatac_snap_path", None)
    elif modality == "atac":
        cfg.pop("scrna_data_path", None)
    if TOOLS[modality]:
        cfg["tool_category"] = TOOLS[modality]
    cfg["capture_telemetry"] = False
    cfg["send_images"] = VISION[vision]
    cfg["output_base_dir"] = f"output/htan_ablation/{short}/{arm}/{cancer}/{task}/"
    cfg["llm"] = {"provider": "anthropic", "model": model, "prompt_caching": True, **MODEL_LLM[model]}
    cfg["logging"] = {**(src_cfg.get("logging") or {}), "project": WANDB_PROJECT, "run_name": None}
    cfg["ablation"] = {"model": model, "arm": arm, "modality": modality, "images": VISION[vision],
                       "cancer": cancer, "task": task, "source_config": f"config/htan/opus-4-6/{cancer}/{task}.yaml"}
    return cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cancers", nargs="*", help="cancer types to generate (default: all)")
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(MODEL_LLM), help="model to run")
    ap.add_argument("--check", action="store_true", help="validate rewrites only; write nothing")
    args = ap.parse_args()

    cancers = sorted(p.name for p in SRC_CONFIGS.iterdir() if p.is_dir())
    if args.cancers:
        unknown = set(args.cancers) - set(cancers)
        if unknown:
            sys.exit(f"unknown cancer types: {sorted(unknown)} (have {cancers})")
        cancers = sorted(args.cancers)

    n_cfg, errors, notes = 0, [], []
    for cancer in cancers:
        for src in sorted((SRC_CONFIGS / cancer).glob("*.yaml")):
            task = src.stem
            src_cfg = yaml.safe_load(src.read_text())
            orig_task = REPO / src_cfg["task_file"]
            text = orig_task.read_text()
            if ("scatac_snap_path" in src_cfg) != ("`adata_atac_backed`" in text):
                notes.append(f"{cancer}/{task}: scatac_snap_path {'set' if 'scatac_snap_path' in src_cfg else 'not set'} "
                             f"but prompt {'mentions' if '`adata_atac_backed`' in text else 'does not mention'} "
                             "adata_atac_backed (kept as in the original)")
            for modality in MODALITIES:
                for vision in VISION:
                    try:
                        variant = text if modality == "multiome" else rewrite(text, modality)
                        variant = rewrite_plots(variant, VISION[vision])
                    except RewriteError as e:
                        errors.append(f"{cancer}/{task} [{modality}_{vision}]: {e}")
                        continue
                    task_file = OUT_TASKS / f"{modality}_{vision}" / cancer / f"{task}.md"
                    if not args.check:
                        task_file.parent.mkdir(parents=True, exist_ok=True)
                        task_file.write_text(variant)
                    cfg = make_config(src_cfg, cancer, task, modality, vision, task_file, args.model)
                    out = OUT_CONFIGS / args.model.removeprefix("claude-") / f"{modality}_{vision}" / cancer / f"{task}.yaml"
                    if not args.check:
                        out.parent.mkdir(parents=True, exist_ok=True)
                        out.write_text(f"# Generated by scripts/make_ablation_configs.py from "
                                       f"{cfg['ablation']['source_config']}; do not edit by hand.\n"
                                       + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
                    n_cfg += 1

    for n in notes:
        print("note:", n)
    for e in errors:
        print("ERROR:", e, file=sys.stderr)
    print(f"{'checked' if args.check else 'wrote'} {n_cfg} configs for {len(cancers)} cancer type(s)"
          + (f"; {len(errors)} rewrite error(s)" if errors else ""))
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
