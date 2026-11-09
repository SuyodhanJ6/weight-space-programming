#!/usr/bin/env python3
"""wsp_train.py - name requested by the WSP-001 goal for the experiment runner.

NOTE: despite the name this is NOT a training script. There is no loss.backward(), optimizer,
or gradient step anywhere. It simply delegates to src/experiment_runner.py (init | run | status | final),
which executes the hypothesis-driven weight-programming loop via src/wsp_engine.py.
"""
import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    runpy.run_module("experiment_runner", run_name="__main__")
