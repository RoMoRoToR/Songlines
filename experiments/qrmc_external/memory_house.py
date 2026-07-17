"""MemoryHouse — a framework-agnostic memory task with a Q/R/M/C
tool-trace contract.

The agent (whatever framework runs it) gets four tools over an external
episodic memory of a household and must find and take an item:

    recall(query)   -> matching place records from memory
    goto(place_id)  -> move to a place, observe its contents
    take(item)      -> pick the item up (must be at the right place)
    finish()        -> end the episode

The environment logs every tool call, so the Q/R/M/C events are
computed identically for every framework:

    Q* -- the agent queried its memory (>=1 recall call);
    R* -- some recall result contained the record of the TRUE location;
    M* -- the agent committed to the true place (goto(true_place));
    C* -- take succeeded within the tool budget.

Task variants engineer exactly one failing stage each:

    control      -- one clean record, generous budget (all stages pass);
    r_starved    -- the location fact was never consolidated into
                    memory: no record mentions the item (R fails);
    m_ambiguous  -- two records mention the item; the stale one (marked
                    as an old note) is listed first; only one place
                    really has it (M degrades);
    c_budget     -- clean memory, budget of 2 tool calls: recall+goto
                    fit, take does not (C fails).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOMS = ["kitchen", "garage", "bedroom", "study", "hallway", "bathroom"]
ITEMS = ["batteries", "scissors", "flashlight", "charger", "tape",
         "matches", "screwdriver", "notebook"]
FILLER = ["old magazines", "a broken umbrella", "spare buttons",
          "a dusty vase", "extra towels", "paper clips"]

VARIANTS = ["control", "r_starved", "m_ambiguous", "c_budget"]
BUDGETS = {"control": 6, "r_starved": 6, "m_ambiguous": 6, "c_budget": 2}

# ── fault taxonomy for the blinded localization meta-evaluation ──────
# Each fault is injected into exactly one stage-owning component; the
# injected stage is the ground-truth label the blinded decision rule
# must recover from the Q/R/M/C profile alone.
#
#   q_no_tool     Q  the recall tool is not offered to the agent;
#   q_misleading  Q  prompt-level: the task text tells the agent its
#                    memory is from previous tenants (query formation
#                    is discouraged, memory itself is intact);
#   r_starved     R  (existing) the fact was never consolidated;
#   r_corrupted   R  silent corruption: the true record omits the item
#                    and a confident record names it at a wrong place;
#   r_flaky       R  recall fails with p=0.7 per call ("memory
#                    temporarily unavailable");
#   m_ambiguous   M  (existing) stale distractor listed first;
#   m_duplicate   M  two equally fresh records name the item at two
#                    places (wrong one first), no staleness marker;
#   c_budget      C  (existing) budget of 2 tool calls;
#   c_flaky_goto  C  goto fails to move with p=0.5 ("door is stuck");
#                    the commitment is logged, arrival fails;
#   c_take_broken C  take always fails ("your hands are full").
FAULT_VARIANTS = [
    "q_no_tool", "q_misleading", "r_corrupted", "r_flaky",
    "m_duplicate", "c_flaky_goto", "c_take_broken",
]
ALL_VARIANTS = VARIANTS + FAULT_VARIANTS
BUDGETS.update({v: 6 for v in FAULT_VARIANTS})

# ground-truth stage per variant (None = no fault injected)
FAULT_STAGE = {
    "control": "none",
    "q_no_tool": "Q", "q_misleading": "Q",
    "r_starved": "R", "r_corrupted": "R", "r_flaky": "R",
    "m_ambiguous": "M", "m_duplicate": "M",
    "c_budget": "C", "c_flaky_goto": "C", "c_take_broken": "C",
}
P_RECALL_FAIL = 0.7   # r_flaky
P_GOTO_STUCK = 0.5    # c_flaky_goto


class MemoryHouse:
    def __init__(self, variant: str, seed: int,
                 repair: "Optional[str]" = None):
        # repair in {None, "Q", "R", "M", "C"}: a stage-level repair
        # from the fixed menu (fix query channel / consolidate memory /
        # remove untrue item records / relax execution), applied
        # without knowledge of which fault was injected.
        assert variant in ALL_VARIANTS
        assert repair in (None, "Q", "R", "M", "C")
        rng = np.random.default_rng(7000 + seed)
        self.variant = variant
        self.repair = repair
        self.budget = BUDGETS[variant] + (4 if repair == "C" else 0)
        self.item = ITEMS[int(rng.integers(0, len(ITEMS)))]
        # separate stream for runtime fault coin flips, deterministic
        # per seed and independent of layout generation
        self._frng = np.random.default_rng(9000 + seed)

        rooms = list(rng.permutation(ROOMS))
        self.places: Dict[str, Dict[str, Any]] = {}
        for k, room in enumerate(rooms[:4]):
            pid = f"p{k+1}"
            self.places[pid] = {"id": pid, "room": room,
                                "contents": [FILLER[int(rng.integers(0, len(FILLER)))]]}

        true_pid = f"p{int(rng.integers(1, 5))}"
        self.true_place = true_pid
        self.places[true_pid]["contents"].append(self.item)

        # memory records the agent can recall (the "prior session")
        self.records: List[Dict[str, str]] = []
        for pid, pl in self.places.items():
            note = f"{pl['room']} ({pid}): {', '.join(pl['contents'])}"
            if variant in ("r_starved", "r_corrupted") and pid == true_pid:
                # the fact was never consolidated: record omits the item
                note = (f"{pl['room']} ({pid}): "
                        f"{', '.join(c for c in pl['contents'] if c != self.item)}")
            self.records.append({"place_id": pid, "note": note})
        if variant == "m_ambiguous":
            # a stale note pointing at a wrong place, listed FIRST
            wrong = [p for p in self.places if p != true_pid]
            self.stale_place = wrong[int(rng.integers(0, len(wrong)))]
            stale_note = {"place_id": self.stale_place,
                          "note": (f"{self.places[self.stale_place]['room']} "
                                   f"({self.stale_place}): {self.item} "
                                   f"(old note, may be outdated)")}
            self.records.insert(0, stale_note)
        if variant == "r_corrupted":
            # silent corruption: a confident record names the item at a
            # wrong place; no staleness marker, the true record omits it
            wrong = [p for p in self.places if p != true_pid]
            self.decoy_place = wrong[int(rng.integers(0, len(wrong)))]
            self.records.insert(0, {
                "place_id": self.decoy_place,
                "note": (f"{self.places[self.decoy_place]['room']} "
                         f"({self.decoy_place}): {self.item}")})
        if variant == "m_duplicate":
            # two equally fresh records name the item; wrong one FIRST
            wrong = [p for p in self.places if p != true_pid]
            self.decoy_place = wrong[int(rng.integers(0, len(wrong)))]
            self.records.insert(0, {
                "place_id": self.decoy_place,
                "note": (f"{self.places[self.decoy_place]['room']} "
                         f"({self.decoy_place}): {self.item}")})

        if repair == "R":
            # consolidate: the true record names the item
            for r in self.records:
                if r["place_id"] == true_pid and self.item not in r["note"].lower():
                    r["note"] = (f"{self.places[true_pid]['room']} "
                                 f"({true_pid}): "
                                 f"{', '.join(self.places[true_pid]['contents'])}")
        if repair == "M":
            # remove records that name the item at a non-true place
            self.records = [r for r in self.records
                            if not (self.item in r["note"].lower()
                                    and r["place_id"] != true_pid)]

        # runtime state
        self.at: Optional[str] = None
        self.tool_calls = 0
        self.done = False
        self.log: List[Dict[str, Any]] = []

    # ── tools ─────────────────────────────────────────────────────

    def _spend(self, tool: str, arg: str) -> Optional[str]:
        if self.done:
            return "Episode is over."
        if self.tool_calls >= self.budget:
            self.done = True
            return "Budget exhausted; the episode is over."
        self.tool_calls += 1
        self.log.append({"tool": tool, "arg": arg, "n": self.tool_calls})
        return None

    def recall(self, query: str) -> str:
        stop = self._spend("recall", query)
        if stop:
            return stop
        if (self.variant == "r_flaky" and self.repair != "R"
                and self._frng.random() < P_RECALL_FAIL):
            self.log[-1]["hits"] = []
            self.log[-1]["attributed_hits"] = []
            return "Memory is temporarily unavailable. Try again later."
        q = (query or "").lower()
        hits = [r for r in self.records
                if any(tok and tok in r["note"].lower()
                       for tok in q.replace(",", " ").split())] or self.records
        self.log[-1]["hits"] = [r["place_id"] for r in hits]
        self.log[-1]["attributed_hits"] = [
            r["place_id"] for r in hits if self.item in r["note"].lower()]
        return "Memory results:\n" + "\n".join(f"- {r['note']}" for r in hits)

    def _resolve_place(self, s: str) -> Optional[str]:
        s = (s or "").strip().lower()
        for p in self.places:
            if p in s:
                return p
        for p, pl in self.places.items():
            if pl["room"] in s:
                return p
        return None

    def goto(self, place_id: str) -> str:
        stop = self._spend("goto", place_id)
        if stop:
            return stop
        pid = self._resolve_place(place_id)
        self.log[-1]["resolved"] = pid or ""
        if pid is None:
            return ("Unknown place. Valid ids: "
                    + ", ".join(sorted(self.places)) + ".")
        if (self.variant == "c_flaky_goto" and self.repair != "C"
                and self._frng.random() < P_GOTO_STUCK):
            # the commitment is made (resolved is logged above); the
            # move itself fails, so arrival --- not the lock --- breaks
            pl = self.places[pid]
            return (f"The door to {pl['room']} ({pid}) is stuck; "
                    f"you could not enter.")
        self.at = pid
        pl = self.places[pid]
        return (f"You are now at {pl['room']} ({pid}). "
                f"You see: {', '.join(pl['contents'])}.")

    def take(self, item: str) -> str:
        stop = self._spend("take", item)
        if stop:
            return stop
        if self.variant == "c_take_broken" and self.repair != "C":
            return ("You cannot pick that up right now "
                    "(your hands are full).")
        it = (item or "").strip().lower()
        if (self.at == self.true_place and self.item in it):
            self.done = True
            self.log[-1]["success"] = True
            return f"You picked up the {self.item}. Task complete."
        return "That item is not here."

    def finish(self) -> str:
        self.done = True
        return "Episode finished."

    # ── Q/R/M/C from the trace ───────────────────────────────────

    def qrmc(self) -> Dict[str, int]:
        recalls = [e for e in self.log if e["tool"] == "recall"]
        q = int(len(recalls) > 0)
        # R*: attribution semantics (the grid analogue of 'a returned
        # candidate is a real water cell'): some recall result contains
        # a record that MENTIONS the item AND points at the true place.
        r = int(any(self.true_place in e.get("attributed_hits", [])
                    for e in recalls))
        # M*: episode-level, faithful to Definition 1 of the framework
        # (the planner locked a real target at SOME point).  The first
        # commitment (M1) and the stale-first pull are auxiliary
        # diagnostics, exactly as lock-quality annotations are in the
        # grid experiments.
        gotos = [e for e in self.log
                 if e["tool"] == "goto" and e.get("resolved")]
        m = int(any(e["resolved"] == self.true_place for e in gotos))
        m1 = int(bool(gotos) and gotos[0]["resolved"] == self.true_place)
        stale_first = int(bool(gotos) and self.variant == "m_ambiguous"
                          and gotos[0]["resolved"] == self.stale_place)
        c = int(any(e.get("success") for e in self.log))
        exhausted = int(self.tool_calls >= self.budget and not c)
        return {"Q": q, "R": r, "M": m, "C": c, "M1": m1,
                "stale_first": stale_first, "exhausted": exhausted,
                "n_tool_calls": self.tool_calls}

    @property
    def task_text(self) -> str:
        if self.repair == "Q":
            return (f"Find and take the {self.item}. You have a budget of "
                    f"{self.budget} tool calls. First search your memory "
                    f"of the house with recall, then goto the right "
                    f"place, then take the item.")
        if self.variant == "q_no_tool":
            # no recall tool is offered, so the instruction to use it
            # is dropped; everything else is identical
            return (f"Find and take the {self.item}. You have a budget "
                    f"of {self.budget} tool calls. Goto places to look "
                    f"around, then take the item.")
        if self.variant == "q_misleading":
            return (f"Find and take the {self.item}. You have a budget "
                    f"of {self.budget} tool calls. Note: your episodic "
                    f"memory is from the previous tenants of a different "
                    f"house and is probably unreliable. Goto places to "
                    f"look around, then take the item.")
        return (f"Find and take the {self.item}. You have a budget of "
                f"{self.budget} tool calls. First search your memory of "
                f"the house with recall, then goto the right place, then "
                f"take the item.")

    @property
    def available_tools(self) -> List[str]:
        if self.variant == "q_no_tool" and self.repair != "Q":
            return ["goto", "take", "finish"]
        return ["recall", "goto", "take", "finish"]

    def tool_specs(self) -> List[Dict[str, Any]]:
        def spec(name, desc, pname, pdesc):
            return {"type": "function", "function": {
                "name": name, "description": desc,
                "parameters": {"type": "object", "properties": {
                    pname: {"type": "string", "description": pdesc}},
                    "required": [pname]}}}
        specs = [
            spec("recall", "Search your episodic memory of the house.",
                 "query", "what to search for"),
            spec("goto", "Move to a place by its id (e.g. p2).",
                 "place_id", "place id such as p1..p5"),
            spec("take", "Take an item at your current place.",
                 "item", "item name"),
            {"type": "function", "function": {
                "name": "finish", "description": "End the episode.",
                "parameters": {"type": "object", "properties": {}}}},
        ]
        avail = set(self.available_tools)
        return [s for s in specs if s["function"]["name"] in avail]

    def call(self, name: str, args: Dict[str, Any]) -> str:
        if name == "recall":
            return self.recall(str(args.get("query", "")))
        if name == "goto":
            return self.goto(str(args.get("place_id", args.get("place", ""))))
        if name == "take":
            return self.take(str(args.get("item", "")))
        if name == "finish":
            return self.finish()
        return "Unknown tool."
