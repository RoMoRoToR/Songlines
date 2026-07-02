"""LLM-collective bridge — Phase A (Minimal) and Phase B (Standard).

This package proves Q/R/M/C-Diag's measurement interface is agent-class
agnostic: the same Q/R/M/C logger that ran on symbolic planners and
RL CommNet baselines emits non-trivial events on LLM-driven agents,
without modification to experiments/big_experiment/runner.py.

The submodules implement only the agent-side adapters (tag extraction,
query formation, decision) and a thin ALFWorld text-environment
adapter. The Q/R/M/C measurement layer is unchanged.
"""
