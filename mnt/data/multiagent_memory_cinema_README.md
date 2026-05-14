# Collective Memory Cinema Experiment

This visual experiment renders, on the same location:

1. the world state,
2. scout-A local memory graph,
3. scout-B local memory graph,
4. consumer-C local memory graph,
5. the shared concept memory over the same coordinates,
6. a small table with semantic field activations.

It is designed as a **visual demonstrator** for the collective-memory stack:
local memories -> shared concepts -> semantic field.

## Run

```bash
python3 /mnt/data/multiagent_memory_cinema_experiment.py \
  --out_dir /mnt/data/memory_cinema_demo \
  --steps 11
```

## Outputs

- `frames/frame_*.png` — step-by-step frames
- `summary.json` — final summary
- `history.json` — concept history over time

## What to look at

- local graph nodes grow independently for each agent;
- water / hazard semantics appear first locally;
- shared concepts are drawn on the same map with centroid circles;
- the consumer is linked by a dashed line to the currently strongest shared water concept;
- field activations show why one shared concept wins over another.
