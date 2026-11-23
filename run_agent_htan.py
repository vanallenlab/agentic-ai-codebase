#!/usr/bin/env python3
"""Scientific Data Analysis Agent — HTAN batch entry point.

Thin wrapper around run_agent.py so both entry points share one (hardened)
agent loop. HTAN-specific inputs are plain config keys handled by run_agent:
    cnmf_usage_path, cnmf_spectra_scores_path, cnmf_top_genes_path,
    capture_telemetry
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from run_agent import main, run_agent  # noqa: E402,F401

if __name__ == "__main__":
    main()
