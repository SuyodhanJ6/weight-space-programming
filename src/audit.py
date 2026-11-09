"""Alias for weight_audit (either name works)."""
from weight_audit import run_audit  # noqa: F401

if __name__ == "__main__":
    import runpy
    runpy.run_module("weight_audit", run_name="__main__")
