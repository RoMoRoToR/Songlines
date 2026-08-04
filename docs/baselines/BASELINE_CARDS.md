# Baseline cards (Stage 10)

Four direct baseline classes for the unified equal-budget benchmark
(`experiments/song_grammar/baselines.py`, driver `exp_b_unified.py`).
Each card follows the plan's required fields. **Offline caveat:** the
original repositories are not fetched in this environment; these are
faithful mechanism re-implementations from the papers' descriptions,
held to the same episode stream, budgets and metrics as every other
arm. None is a deliberately weakened strawman — where a mechanism is
simplified, it is stated and the simplification does not touch the
axis being tested.

Every baseline consumes the SAME world (`World`, two intents, global
drift, private frames), the SAME walker/cost (`walk`), and the SAME
paired test seeds (100+, never used to tune anything).

---

## Baseline 1 — decision_centric (DeMem-style)
- **Source:** *Remember the Decision, Not the Description: A Rate-Distortion Framework for Agent Memory* (arXiv:2605.10870).
- **Implemented:** episodes merge only when the optimal decision barely changes — operationalised as agreement of the songs' end-to-end displacement within `d_thr=3` (our decision-distortion `D`); shared raw; consumed newest-first.
- **Changed for benchmark:** decision distortion uses the same displacement metric as the Songlines analogy so the two are directly comparable; no other change.
- **Not applicable / omitted:** the paper's rate-distortion optimiser over a continuous description space — our substrate has a discrete song codec, so distortion is exact, not variational.
- **Budget:** same wire/memory accounting as all arms (song codec + timestamp).
- **What it isolates:** does coordination-aware analogy + receiver admission beat individual decision-preserving compression? (DeMem has neither.)
- **Why fair:** it gets the strongest form of its own idea (exact decision distortion) and the same everything else.

## Baseline 2 — execution_path (Mage-style)
- **Source:** *Beyond Semantic Organization: Memory as Execution State Management for Long-Horizon Agents* (arXiv:2606.06090).
- **Implemented:** each agent stores its own execution path per family with revise (latest execution state wins) and rollback; consumed by path similarity.
- **Changed:** a received path is applied at face value (owner's frame) because Mage is single-agent — this is the honest consequence of transplanting a single-agent method into a private-frame collective, not a nerf.
- **Not applicable / omitted:** Mage's LLM-workflow specifics; we keep the execution-state-path core.
- **Budget:** identical accounting.
- **What it isolates:** does distributed route provenance + meaning-based cross-frame identity beat single-agent path saving?
- **Why fair:** it keeps revise/rollback (its strengths) and full paths; only the cross-frame identity it never had is absent.

## Baseline 3 — graph_memory (Generative-Agents / graph-vector)
- **Source:** Generative Agents recency×importance×relevance (arXiv:2304.03442); agent-native memory survey (arXiv:2606.24775).
- **Implemented:** signature graph with RIR retrieval (0.4 recency + 0.3 importance + 0.3 relevance), summary dedup at cosine ≥ 0.9, top-3 consumption; shared raw.
- **Changed:** importance is a simple visit-reinforced scalar (no LLM reflection step); relevance is signature-bag cosine.
- **Not applicable / omitted:** LLM-authored reflections/summaries — replaced by deterministic dedup so the comparison is compute-fair.
- **Budget:** identical accounting.
- **What it isolates:** is the gain graph structure in general, or the Songlines contract (provenance + admission + world-clock) specifically?
- **Why fair:** it gets the full RIR scoring that is its actual contribution; only the orthogonal LLM-summary flourish is deterministic.

## Baseline 4 — learned_formation (Mem-α-style, stripped)
- **Source:** *Mem-α: Learning Memory Construction via Reinforcement Learning* (arXiv:2509.25911).
- **Implemented:** a learned store/merge/drop controller over the same features as our U2 bandit, WITHOUT explicit EXCEPTION, provenance, or receiver-side admission; shared raw.
- **Changed:** the learned policy is the frozen U2-style rule (utility + share gates); a fully trained LinUCB can be dropped in via the `bandit` arg.
- **Not applicable / omitted:** QA-reward training loop of the original — replaced by the coordination-cost reward used throughout the series.
- **Budget:** identical accounting.
- **What it isolates:** are the explicit EXCEPTION / provenance / admission operations necessary, or does a learned formation policy suffice?
- **Why fair:** it is exactly our own learned controller minus the three contract operations — the sharpest possible test of their necessity.

---
**Verdict location:** `docs/CLAIM_EVIDENCE_MATRIX.md` (rows to be added after the cluster benchmark), main table in `exp_b_unified` output. **Acceptance:** none of these is a strawman; each keeps the strength of its own idea and is stripped only of the axis under test.
