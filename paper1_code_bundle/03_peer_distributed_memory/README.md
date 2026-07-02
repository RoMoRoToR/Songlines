# 03 - Peer / Distributed Memory

This folder contains the memory-sharing implementations used for independent,
peer-broadcast, and distributed-consensus variants.

## Core Modules

- `peer_memory/` - peer-to-peer snapshot/broadcast memory.
- `distributed_memory/` - distributed agent memory and consensus layer.
- `independent_memory/` - private-memory baseline.

## Example Experiments

- `peer_memory/exp01_basic_broadcast.py`
- `peer_memory/exp02_asymmetric_trust.py`
- `peer_memory/exp03_three_way_ablation.py`
- `distributed_memory/exp01_basic_per_agent.py`
- `distributed_memory/exp02_consensus_alignment.py`
- `distributed_memory/exp03_disagreement.py`
- `distributed_memory/exp04_trust_weighted_fusion.py`
- `distributed_memory/exp05_partial_observability.py`
- `independent_memory/exp01_isolation.py`

## Original Locations

These files were copied from top-level `peer_memory/`, `distributed_memory/`,
`independent_memory/`, and the corresponding `experiments/` subfolders.

