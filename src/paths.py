"""Experiment namespace. wsp-001 (default) keeps its original paths untouched;
wsp-002 / wsp-003 (WSP_EXPERIMENT=...) each have fully separate state, checkpoints, logs and reports.
"""

from __future__ import annotations

import os
from pathlib import Path

EXPERIMENT = os.environ.get("WSP_EXPERIMENT", "wsp-001")
if EXPERIMENT not in ("wsp-001", "wsp-002", "wsp-003", "wsp-004", "wsp-005", "wsp-006"):
    raise ValueError(f"unknown experiment {EXPERIMENT}")
LOO_RULE = EXPERIMENT == "wsp-006"        # rule v3: pooled leave-one-out over all 16 train+val images
IS_V2 = EXPERIMENT != "wsp-001"          # "v2" = large-split protocol (wsp-002, wsp-003)
TAG = EXPERIMENT.split("-")[1]

EXP = Path(f"experiments/{EXPERIMENT}") if IS_V2 else Path("experiments")
CHECKPOINT_DIR = Path(f"checkpoints/{EXPERIMENT}") if IS_V2 else Path("checkpoints")
LOG_DIR = Path(f"logs/{EXPERIMENT}") if IS_V2 else Path("logs")
REPORT_DIR = Path(f"reports/{EXPERIMENT}") if IS_V2 else Path("reports")
FINAL_REPORT = Path(f"reports/WSP_{TAG}_FINAL_REPORT.md") if IS_V2 else Path("reports/FINAL_REPORT.md")
BASELINE_CKPT = str(CHECKPOINT_DIR / "iter000_initial.pt")
ITERATIONS_MODULE = f"iterations_wsp{TAG}" if IS_V2 else "iterations"

DATA_DIR = Path(f"dataset/{EXPERIMENT}") if IS_V2 else Path("dataset/wsp-002")
V2_IMAGE_DIR = DATA_DIR / "images"          # train/ and val/ (readable)
V2_SEALED_DIR = DATA_DIR / "sealed_test"    # chmod 000 until the single final evaluation
V2_CACHE_DIR = DATA_DIR / "cache"
V2_MANIFEST = EXP / "manifest.json"
HASH_FILE = EXP / ("wsp001_artifact_hashes.sha256" if EXPERIMENT == "wsp-002"
                   else "previous_artifact_hashes.sha256")
