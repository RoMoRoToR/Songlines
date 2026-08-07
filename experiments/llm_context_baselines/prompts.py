"""Prompt library for the LLM context-baseline suite (package H).

Design discipline: every arm answers the SAME questions under the SAME
system prompts with the SAME scoring; only the MEMORY PAYLOAD section
of the user prompt differs between arms.  The freshness system prompt
and question line are imported from the original L1 experiment
(experiments/song_grammar/exp_l1_llm_endtoend.py) verbatim, so the raw
and graph arms replicate the published protocol exactly.
"""

from __future__ import annotations

# The published L1 system prompt (freshness question).  Imported, not
# copied, so it can never drift from the original experiment.
from experiments.song_grammar.exp_l1_llm_endtoend import SYSTEM as SYSTEM_FRESHNESS

# Identity question: how many distinct water sites exist across the
# whole history (appearance variants differ per visit; the same site
# must be recognised as one place, different sites as different).
SYSTEM_IDENTITY = (
    "You are a navigation agent reviewing your memory of a grid world. "
    "Coordinates identify places; visual appearance varies between "
    "visits, so two visits reaching the same coordinates are the SAME "
    "place and different coordinates are DIFFERENT places. Count how "
    "many DISTINCT water-source locations (unique coordinates) appear "
    "in your memory across the ENTIRE history, current and stale "
    "together. Answer with a single JSON object exactly like "
    "{\"count\": n} and nothing else.")

# Question lines (identical for every arm).
QUESTION_FRESHNESS = "Where is the water NOW? Answer with JSON."
QUESTION_IDENTITY = (
    "How many DISTINCT water locations (unique coordinates) appear "
    "across the FULL history, current and stale together? "
    "Answer with JSON.")

# ── arm 2: raw transcript + explicit conflict-resolution instructions
RAW_INSTRUCTED_PREAMBLE = (
    "HOW TO READ THE MEMORY BELOW: it is a chronological log of "
    "MULTIPLE visits to a world that CHANGES over time. A resource "
    "may have MOVED between visits, so the log can contain "
    "conflicting reports about the same resource. Resolve conflicts "
    "by RECENCY: a later episode supersedes every earlier one. "
    "Repetition is NOT evidence of currency -- many stale mentions "
    "do not outweigh a single more recent observation. Landmark "
    "descriptions vary between visits (appearance changes); treat "
    "visits that reach the same coordinates as the same place. "
    "First find the LATEST episode(s), then answer from those.")

# ── arm 3: rolling summary (incremental, same LLM as the answerer)
SUMMARIZER_SYSTEM = (
    "You maintain a running MEMORY SUMMARY for a navigation agent in "
    "a grid world. You will be given the current summary and ONE new "
    "episode transcript. Output the UPDATED summary and nothing else "
    "(no preamble, no commentary). The FIRST line must state the most "
    "recently confirmed water location. Also preserve: any evidence "
    "that earlier water sites are stale or superseded, and the list "
    "of distinct water coordinates seen so far with visit counts. "
    "Coordinates matter; landmark appearance details do not.")

SUMMARIZER_TEMPLATE = (
    "CURRENT SUMMARY (may be empty):\n{summary}\n\n"
    "NEW EPISODE:\n{episode}\n\n"
    "Write the updated summary. Hard limit: about {budget_tokens} "
    "tokens (~{budget_chars} characters); anything longer will be "
    "cut off from the end.")

EMPTY_SUMMARY = "(empty -- no episodes summarised yet)"

# ── payload section headers ─────────────────────────────────────────
RAW_HEADER = "FULL MEMORY TRANSCRIPT (chronological):"  # matches L1
RETRIEVAL_HEADER = (
    "RETRIEVED MEMORY (episodes most relevant to the question, "
    "shown in chronological order; the rest of the history did not "
    "fit the budget):")
SUMMARY_HEADER = "MEMORY SUMMARY (incrementally maintained):"
TABLE_HEADER = (
    "MEMORY TABLE (one row per water site ever observed; built "
    "mechanically as latest-state-per-place):\n"
    "site_xy | first_seen_ep | last_seen_ep | n_visits | status")
NO_MEMORY = "You have no stored memory of this world."  # matches L1

TRUNCATION_MARKER = "[... {n} earlier lines dropped to fit the token budget ...]"
