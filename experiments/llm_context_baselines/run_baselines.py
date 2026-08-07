"""H -- LLM context baselines at a FIXED TOKEN BUDGET.

Reviewer claim addressed: "the raw-transcript baseline is too weak;
the graph arm receives an already-solved consolidation -- compare
against modern context-management alternatives at a fixed token
budget."  This suite strengthens the published L1/L1b comparison
(experiments/song_grammar/exp_l1_llm_endtoend.py; paper numbers
raw 0.42->0.08 llama3.1, 1.0->0.375 Qwen2.5-3B, graph 1.0 at 6-12%
payload) with six arms that all receive the SAME evidence stream
(the L1 episode songs), the SAME model, the SAME token budget for
the memory payload, and answer the SAME questions under the SAME
scoring:

  raw       -- chronological transcript, truncated to the budget by a
               RECENCY window (most recent lines kept) -- the standard
               sliding-window practice, strictly stronger than naive
               head truncation on freshness;
  raw_instr -- the same truncated transcript preceded by explicit
               instructions on how to resolve staleness conflicts
               (the preamble COUNTS toward the budget);
  summary   -- rolling summary: the SAME LLM incrementally folds each
               episode into a summary capped at the budget (build
               cost in extra LLM calls/tokens is recorded);
  retrieval -- vector retrieval over per-episode chunks; embedder is
               sentence-transformers if importable, otherwise a
               dependency-free TF-IDF cosine (which one was used is
               recorded in the output -- no silent substitution);
               top chunks are packed into the budget and shown in
               chronological order;
  table     -- structured latest-state-per-place table built
               PROGRAMMATICALLY (no LLM, no schema machinery): the
               cheap structural control the reviewer asks for;
  graph     -- the published Songlines/UCSM active context (schemas +
               certificates + exceptions), unchanged;
  none      -- no-memory control (optional; grounds L1.1).

Questions (identical protocol for every arm):
  freshness -- "Where is the water NOW?" -> {"target":[x,y]};
               scored exactly as L1: lock_correct (target == current
               water) and completed (Dijkstra-reachable WATER cell);
  identity  -- "How many DISTINCT water locations appear across the
               full history?" -> {"count": n}; scored by exact match
               (cross-frame identity: every visit uses a fresh
               appearance variant, sites repeat across episodes).

Token budget: one counter for all arms -- tiktoken (cl100k_base) if
importable, else ceil(chars/4); which one was used is recorded.
Budget 0 means "no cap" (replicates the published unbounded regime).

Usage (local smoke, ollama):
  PYTHONPATH=. python experiments/llm_context_baselines/run_baselines.py \
      --backend ollama --model llama3.1:latest --mode long \
      --layouts 3 --budgets 128 512 --out tmp/llm_context_baselines/smoke

Cluster (hf, see cluster/submit_llm_baselines.sh):
  ... --backend hf --model Qwen/Qwen2.5-3B-Instruct --mode long \
      --layouts 12 --budgets 0 96 192 384 768

Config-grid mode:
  ... --config experiments/llm_context_baselines/configs/full_grid.json \
      --run qwen3b_long
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from experiments.song_grammar.exp_l1_llm_endtoend import (
    ROLE, StubBackend, form_memory, form_memory_long, observation_text,
    parse_target, song_text, ucsm_context)
from experiments.song_grammar.exp_s0_song_smoke import TRAVELER_START
from experiments.song_grammar.u7_common import dijkstra
from multiagent_env import WATER

from experiments.llm_context_baselines import prompts as P

GridXY = Tuple[int, int]
ARMS = ("raw", "raw_instr", "summary", "retrieval", "table", "graph")
QUESTIONS = ("freshness", "identity")
DEFAULT_SUMMARY_TOKENS = 256   # summary target when budget == 0 (no cap)


# ── token counting (one counter for ALL arms) ───────────────────────

def make_token_counter():
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return (lambda s: len(enc.encode(s))), "tiktoken:cl100k_base"
    except Exception:
        return (lambda s: max(1, (len(s) + 3) // 4)), "chars/4"


def truncate_to_budget(text: str, budget: Optional[int], count,
                       keep: str = "tail") -> Tuple[str, int]:
    """Line-granular truncation to <= budget tokens (marker included).
    keep='tail' drops the OLDEST lines (recency window); keep='head'
    drops trailing lines. Returns (text, n_dropped_lines)."""
    if not budget or count(text) <= budget:
        return text, 0
    lines = text.split("\n")
    marker_cost = count(P.TRUNCATION_MARKER.format(n=len(lines))) + 1
    kept: List[str] = []
    used = marker_cost
    seq = reversed(lines) if keep == "tail" else lines
    for ln in seq:
        c = count(ln) + 1
        if used + c > budget:
            break
        kept.append(ln)
        used += c
    if keep == "tail":
        kept.reverse()
    if not kept:
        # a single line exceeds the budget: word-level fallback,
        # sized by binary search on the EXACT token counter
        words = text.split()
        marker = P.TRUNCATION_MARKER.format(n=len(lines))

        def assemble(k: int) -> str:
            sel = words[-k:] if keep == "tail" else words[:k]
            joined = " ".join(sel)
            return (marker + "\n" + joined if keep == "tail"
                    else joined + "\n" + marker)

        lo, hi = 0, len(words)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if count(assemble(mid)) <= budget:
                lo = mid
            else:
                hi = mid - 1
        return assemble(lo), len(lines)
    n_drop = len(lines) - len(kept)
    marker = P.TRUNCATION_MARKER.format(n=n_drop)
    body = kept if keep == "tail" else kept + [marker]
    if keep == "tail":
        body = [marker] + kept
    return "\n".join(body), n_drop


# ── embeddings for the retrieval arm ────────────────────────────────

def _toks(s: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


def tfidf_scores(chunks: List[str], query: str) -> List[float]:
    """Dependency-free TF-IDF cosine (log-tf, smoothed idf)."""
    docs = [_toks(c) for c in chunks]
    df: Dict[str, int] = {}
    for d in docs:
        for w in set(d):
            df[w] = df.get(w, 0) + 1
    n = len(docs)

    def vec(ws: List[str]) -> Dict[str, float]:
        tf: Dict[str, int] = {}
        for w in ws:
            tf[w] = tf.get(w, 0) + 1
        return {w: (1.0 + math.log(c))
                * math.log(1.0 + (n + 1) / (df.get(w, 0) + 1))
                for w, c in tf.items()}

    qv = vec(_toks(query))
    qn = math.sqrt(sum(v * v for v in qv.values()))
    out = []
    for d in docs:
        dv = vec(d)
        dn = math.sqrt(sum(v * v for v in dv.values()))
        dot = sum(qv[w] * dv.get(w, 0.0) for w in qv)
        out.append(dot / (qn * dn) if qn * dn > 0 else 0.0)
    return out


class Embedder:
    """sentence-transformers if importable, else TF-IDF. The choice is
    recorded in every output file -- no silent substitution."""

    def __init__(self, kind: str = "auto"):
        self.kind = "tfidf"
        self._model = None
        if kind in ("auto", "sbert"):
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer("all-MiniLM-L6-v2")
                self.kind = "sbert:all-MiniLM-L6-v2"
            except Exception:
                if kind == "sbert":
                    raise
        if kind == "tfidf":
            self._model = None
            self.kind = "tfidf"

    def scores(self, chunks: List[str], query: str) -> List[float]:
        if self._model is None:
            return tfidf_scores(chunks, query)
        import numpy as np
        em = self._model.encode(chunks + [query], normalize_embeddings=True)
        return list(np.asarray(em[:-1]) @ np.asarray(em[-1]))


# ── payload builders (one per arm) ──────────────────────────────────

def episode_chunks(episodes) -> List[str]:
    return [song_text(s, w, k + 1) for k, (s, w) in enumerate(episodes)]


def build_raw(episodes, budget, count) -> Tuple[str, Dict[str, Any]]:
    body = "\n".join(episode_chunks(episodes))
    body, n_drop = truncate_to_budget(body, budget, count, keep="tail")
    return P.RAW_HEADER + "\n" + body, {"dropped_lines": n_drop}


def build_raw_instr(episodes, budget, count) -> Tuple[str, Dict[str, Any]]:
    pre = P.RAW_INSTRUCTED_PREAMBLE + "\n\n" + P.RAW_HEADER
    inner = budget - count(pre) - 1 if budget else None
    body = "\n".join(episode_chunks(episodes))
    body, n_drop = truncate_to_budget(body, inner, count, keep="tail")
    return pre + "\n" + body, {"dropped_lines": n_drop}


def build_summary(episodes, budget, count, backend, seed,
                  max_tokens) -> Tuple[str, Dict[str, Any]]:
    cap = budget or DEFAULT_SUMMARY_TOKENS
    summary = P.EMPTY_SUMMARY
    calls, in_toks, out_toks = 0, 0, 0
    for k, chunk in enumerate(episode_chunks(episodes)):
        prompt = P.SUMMARIZER_TEMPLATE.format(
            summary=summary, episode=chunk,
            budget_tokens=cap, budget_chars=cap * 4)
        out = backend.complete(prompt, system=P.SUMMARIZER_SYSTEM,
                               seed=seed * 1000 + k,
                               max_tokens=min(max_tokens, cap + 64))
        calls += 1
        in_toks += count(P.SUMMARIZER_SYSTEM) + count(prompt)
        out_toks += count(out)
        summary = out.strip() or summary
        # hard enforcement: keep the HEAD (summariser puts the current
        # water on the first line), drop the tail
        summary, _ = truncate_to_budget(summary, cap, count, keep="head")
    payload = P.SUMMARY_HEADER + "\n" + summary
    payload, _ = truncate_to_budget(payload, budget, count, keep="head")
    return payload, {"build_llm_calls": calls,
                     "build_input_tokens": in_toks,
                     "build_output_tokens": out_toks}


def build_retrieval(episodes, budget, count, embedder, question,
                    ) -> Tuple[str, Dict[str, Any]]:
    chunks = episode_chunks(episodes)
    query = observation_text() + " " + question
    scores = embedder.scores(chunks, query)
    # rank by score, ties -> more recent episode first
    order = sorted(range(len(chunks)),
                   key=lambda i: (-scores[i], -i))
    head_cost = count(P.RETRIEVAL_HEADER) + 1
    used, picked = head_cost, []
    cap = budget or float("inf")
    for i in order:
        c = count(chunks[i]) + 1
        if used + c > cap:
            continue
        picked.append(i)
        used += c
    if not picked and order:      # budget smaller than every chunk:
        i = order[0]              # keep the tail of the best chunk
        body, _ = truncate_to_budget(chunks[i], (budget or 0) - head_cost,
                                     count, keep="tail")
        picked_txt = [body]
        picked = [i]
    else:
        picked.sort()             # chronological presentation
        picked_txt = [chunks[i] for i in picked]
    payload = P.RETRIEVAL_HEADER + "\n" + "\n".join(picked_txt)
    return payload, {"picked_episodes": [i + 1 for i in picked],
                     "embedder": embedder.kind}


def build_table(episodes, budget, count) -> Tuple[str, Dict[str, Any]]:
    """Programmatic latest-state-per-place table. No LLM, no schema
    machinery, no certificates: first/last episode and visit count per
    water site, CURRENT/STALE by most recent observation."""
    stat: Dict[GridXY, List[int]] = {}
    for k, (_s, w) in enumerate(episodes):
        first, last, n = stat.get(tuple(w), [k + 1, k + 1, 0])
        stat[tuple(w)] = [first, k + 1, n + 1]
    latest = max(v[1] for v in stat.values())
    rows = []
    for (x, y), (first, last, n) in sorted(
            stat.items(), key=lambda kv: -kv[1][1]):
        status = "CURRENT" if last == latest else "STALE"
        rows.append(f"({x},{y}) | {first} | {last} | {n} | {status}")
    body = P.TABLE_HEADER + "\n" + "\n".join(rows)
    body, n_drop = truncate_to_budget(body, budget, count, keep="head")
    return body, {"n_sites": len(stat), "dropped_lines": n_drop}


def build_graph(schemas, budget, count) -> Tuple[str, Dict[str, Any]]:
    body = ucsm_context(schemas)
    body, n_drop = truncate_to_budget(body, budget, count, keep="head")
    return body, {"dropped_lines": n_drop}


# ── answering + scoring (identical protocol for every arm) ──────────

def parse_count(text: str) -> Optional[int]:
    for m in reversed(re.findall(r"\{[^{}]*\}", text, re.S)):
        try:
            obj = json.loads(m)
            c = obj.get("count")
            if isinstance(c, (int, float)):
                return int(c)
        except (ValueError, TypeError):
            continue
    ms = re.findall(r"\b(\d+)\b", text)
    return int(ms[-1]) if ms else None


def ask(backend, payload: str, question: str, seed: int,
        max_tokens: int, no_think: bool) -> Tuple[str, str]:
    if question == "freshness":
        system, qline = P.SYSTEM_FRESHNESS, P.QUESTION_FRESHNESS
    else:
        system, qline = P.SYSTEM_IDENTITY, P.QUESTION_IDENTITY
    if no_think:
        system = system + " /no_think"
    prompt = (observation_text() + "\n\n" + payload + "\n\n" + qline)
    out = backend.complete(prompt, system=system, seed=seed,
                           max_tokens=max_tokens)
    return prompt, out


def score_freshness(out: str, formed) -> Dict[str, Any]:
    target = parse_target(out)
    lock = target == tuple(formed["water"])
    completed = False
    if target is not None:
        path, _ = dijkstra(formed["env"], TRAVELER_START, target, ROLE)
        completed = (path is not None
                     and formed["env"].cell(*target) == WATER)
    return {"target": list(target) if target else None,
            "lock_correct": lock, "completed": completed}


def score_identity(out: str, distinct: int) -> Dict[str, Any]:
    c = parse_count(out)
    return {"count": c, "count_correct": c == distinct}


# ── layout loop ─────────────────────────────────────────────────────

def run_layout(backend, embedder, count, fam: int, conflict: bool,
               mode: str, budgets: List[int], arms: List[str],
               seed: int, max_tokens: int, no_think: bool,
               ) -> Optional[Dict[str, Any]]:
    formed = (form_memory_long(fam, conflict) if mode == "long"
              else form_memory(fam, conflict))
    if formed is None:
        return None
    episodes = formed["episodes"]
    distinct = len({tuple(w) for _s, w in episodes})
    raw_full_tokens = count("\n".join(episode_chunks(episodes)))
    row: Dict[str, Any] = {
        "fam": fam, "conflict": conflict, "mode": mode,
        "water": list(formed["water"]), "distinct_waters": distinct,
        "n_episodes": len(episodes),
        "raw_full_tokens": raw_full_tokens, "cells": []}
    for budget in budgets:
        payload_cache: Dict[str, Tuple[str, Dict[str, Any]]] = {}
        for arm in arms:
            for q in QUESTIONS:
                key = arm if arm != "retrieval" else f"{arm}:{q}"
                if key not in payload_cache:
                    if arm == "raw":
                        pl = build_raw(episodes, budget, count)
                    elif arm == "raw_instr":
                        pl = build_raw_instr(episodes, budget, count)
                    elif arm == "summary":
                        pl = build_summary(episodes, budget, count,
                                           backend, seed, max_tokens)
                    elif arm == "retrieval":
                        qline = (P.QUESTION_FRESHNESS if q == "freshness"
                                 else P.QUESTION_IDENTITY)
                        pl = build_retrieval(episodes, budget, count,
                                             embedder, qline)
                    elif arm == "table":
                        pl = build_table(episodes, budget, count)
                    elif arm == "graph":
                        pl = build_graph(formed["schemas"], budget, count)
                    elif arm == "none":
                        pl = (P.NO_MEMORY, {})
                    else:
                        raise ValueError(arm)
                    # hard invariant: NO arm's payload may exceed the
                    # budget (e.g. raw_instr when the preamble alone
                    # nearly fills a tiny budget: instructions survive,
                    # evidence is sacrificed -- the honest cost of
                    # spending budget on instructions)
                    txt, meta0 = pl
                    if budget and count(txt) > budget:
                        keep = ("tail" if arm in ("raw", "retrieval")
                                else "head")
                        txt, extra_drop = truncate_to_budget(
                            txt, budget, count, keep=keep)
                        meta0 = dict(meta0,
                                     final_enforcement_dropped=extra_drop)
                    payload_cache[key] = (txt, meta0)
                payload, meta = payload_cache[key]
                prompt, out = ask(backend, payload, q, seed,
                                  max_tokens, no_think)
                cell = {"arm": arm, "question": q, "budget": budget,
                        "payload_tokens": count(payload),
                        "prompt_tokens": count(prompt),
                        "meta": meta, "raw_reply": out[-160:]}
                if q == "freshness":
                    cell.update(score_freshness(out, formed))
                else:
                    cell.update(score_identity(out, distinct))
                row["cells"].append(cell)
    return row


# ── aggregation ─────────────────────────────────────────────────────

def aggregate(rows: List[Dict[str, Any]], arms: List[str],
              budgets: List[int]) -> Dict[str, Any]:
    def cells(arm, q, budget, conflict=None):
        out = []
        for r in rows:
            if conflict is not None and r["conflict"] != conflict:
                continue
            out.extend(c for c in r["cells"]
                       if c["arm"] == arm and c["question"] == q
                       and c["budget"] == budget)
        return out

    def rate(cs, field):
        return (sum(bool(c[field]) for c in cs) / len(cs)) if cs else None

    summ: Dict[str, Any] = {}
    for budget in budgets:
        b: Dict[str, Any] = {}
        for arm in arms:
            fr = cells(arm, "freshness", budget)
            idn = cells(arm, "identity", budget)
            b[arm] = {
                "freshness_lock": rate(fr, "lock_correct"),
                "freshness_lock_conflict": rate(
                    cells(arm, "freshness", budget, True), "lock_correct"),
                "freshness_lock_base": rate(
                    cells(arm, "freshness", budget, False), "lock_correct"),
                "freshness_completed": rate(fr, "completed"),
                "identity_acc": rate(idn, "count_correct"),
                "mean_payload_tokens": (
                    sum(c["payload_tokens"] for c in fr) / len(fr)
                    if fr else None),
                "build_llm_calls": sum(
                    c["meta"].get("build_llm_calls", 0) for c in fr),
            }
        summ[f"budget_{budget}"] = b
    return summ


def print_table(summ: Dict[str, Any], arms: List[str]) -> str:
    lines = []
    for bkey, b in summ.items():
        lines.append(f"\n== {bkey} (0 = no cap) ==")
        lines.append(f"{'arm':>10} | {'fresh':>5} {'confl':>5} "
                     f"{'base':>5} | {'ident':>5} | {'tokens':>7} "
                     f"| {'bcalls':>6}")
        for arm in arms:
            a = b[arm]

            def f(v):
                return "  -  " if v is None else f"{v:5.2f}"

            tok = a["mean_payload_tokens"]
            lines.append(
                f"{arm:>10} | {f(a['freshness_lock'])} "
                f"{f(a['freshness_lock_conflict'])} "
                f"{f(a['freshness_lock_base'])} | "
                f"{f(a['identity_acc'])} | "
                f"{(tok if tok is None else int(tok)) or 0:>7} "
                f"| {a['build_llm_calls']:>6}")
    txt = "\n".join(lines)
    print(txt, flush=True)
    return txt


# ── main ────────────────────────────────────────────────────────────

def make_backend(a):
    if a.backend == "hf":
        from experiments.llm_collective.hf_backend import HFBackend
        return HFBackend(model=a.model,
                         cache_dir=os.path.join(a.out, ".cache"))
    if a.backend == "ollama":
        from experiments.llm_collective.llm_backend import OllamaBackend
        return OllamaBackend(
            model=a.model, cache_dir=os.path.join(a.out, ".cache"),
            timeout_s=float(os.environ.get("OLLAMA_TIMEOUT", "300")))
    return StubBackend()


def execute(a) -> None:
    os.makedirs(a.out, exist_ok=True)
    count, counter_name = make_token_counter()
    embedder = Embedder(a.embedder)
    backend = make_backend(a)
    arms = list(a.arms)
    if a.include_none and "none" not in arms:
        arms = ["none"] + arms

    with open(os.path.join(a.out, "registered.json"), "w") as f:
        json.dump({
            "package": "H llm_context_baselines",
            "reviewer_claim": "raw-transcript baseline too weak; graph "
                              "arm receives solved consolidation; compare "
                              "vs modern alternatives at fixed budget",
            "protocol": "same model, same evidence episodes, same token "
                        "budget per memory payload, same questions "
                        "(freshness = published L1 question verbatim; "
                        "identity = distinct-site count), same scoring",
            "token_counter": counter_name,
            "embedder": embedder.kind,
            "arms": arms, "budgets": a.budgets, "mode": a.mode,
            "measured": [
                "H.A: budget-matched ranking of graph vs raw/raw_instr/"
                "summary/retrieval on conflict freshness",
                "H.B: whether the programmatic table (cheap structural "
                "control) closes the raw->graph gap (reported either way)",
                "H.C: build-cost accounting (extra LLM calls/tokens) for "
                "the summary arm vs the deterministic graph/table builders",
            ],
        }, f, indent=2)

    rows: List[Dict[str, Any]] = []
    fam, tries = a.fam0, 0
    while len(rows) < a.layouts and tries < a.layouts * 20:
        tries += 1
        fam += 1
        conflict = (len(rows) % 3 != 0 if a.mode == "long"
                    else len(rows) % 2 == 1)
        row = run_layout(backend, embedder, count, fam, conflict,
                         a.mode, a.budgets, arms, seed=len(rows),
                         max_tokens=a.max_tokens, no_think=a.no_think)
        if row is not None:
            rows.append(row)
            got = [c for c in row["cells"] if c["question"] == "freshness"
                   and c["budget"] == a.budgets[-1]]
            print(f"layout {len(rows)} (conflict={conflict}, "
                  f"raw={row['raw_full_tokens']}tok): "
                  + " ".join(f"{c['arm']}:"
                             f"{'OK' if c['lock_correct'] else 'x'}"
                             for c in got), flush=True)
    with open(os.path.join(a.out, "rows.json"), "w") as f:
        json.dump(rows, f, indent=1)

    summ = aggregate(rows, arms, a.budgets)
    table = print_table(summ, arms)
    result = {
        "model": backend.summary() if hasattr(backend, "summary")
        else {"model": a.backend},
        "backend": a.backend, "mode": a.mode,
        "n_layouts": len(rows), "budgets": a.budgets,
        "token_counter": counter_name, "embedder": embedder.kind,
        "mean_raw_full_tokens": (sum(r["raw_full_tokens"] for r in rows)
                                 / max(1, len(rows))),
        "summary": summ,
    }
    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(a.out, "table.txt"), "w") as f:
        f.write(table + "\n")
    if hasattr(backend, "close"):
        backend.close()
    print(f"Saved: {a.out}/results.json")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["stub", "ollama", "hf"],
                    default="stub")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--mode", choices=["short", "long"], default="long")
    ap.add_argument("--layouts", type=int, default=12)
    ap.add_argument("--budgets", nargs="+", type=int,
                    default=[0, 96, 192, 384, 768],
                    help="memory-payload token budgets; 0 = no cap")
    ap.add_argument("--arms", nargs="+", default=list(ARMS),
                    choices=list(ARMS) + ["none"])
    ap.add_argument("--include-none", action="store_true", default=True)
    ap.add_argument("--no-include-none", dest="include_none",
                    action="store_false")
    ap.add_argument("--embedder", choices=["auto", "tfidf", "sbert"],
                    default="auto")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--fam0", type=int, default=5000,
                    help="layout family offset (5000 matches L1)")
    ap.add_argument("--out", default="tmp/llm_context_baselines/dev")
    ap.add_argument("--config", default=None,
                    help="JSON config with a 'runs' grid")
    ap.add_argument("--run", default="all",
                    help="run name from the config grid, or 'all'")
    return ap


def main() -> None:
    ap = build_parser()
    a = ap.parse_args()
    if a.config:
        with open(a.config) as f:
            grid = json.load(f)
        for spec in grid["runs"]:
            if a.run not in ("all", spec["name"]):
                continue
            merged = ap.parse_args([])       # defaults
            for k, v in spec.items():
                if k != "name":
                    setattr(merged, k, v)
            print(f"\n#### run: {spec['name']} -> {merged.out}",
                  flush=True)
            execute(merged)
    else:
        execute(a)


if __name__ == "__main__":
    main()
