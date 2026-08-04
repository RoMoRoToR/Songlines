"""
Structural phase diagram of the communication category: trust x cadence.

Program step: parameters -> categorical invariants. We model not the memory but
the COMMUNICATION STRUCTURE, and read off when a collective consensus object
(a colimit spanning all agents) exists.

Two knobs, two structural roles (forwarded reframing):
  - trust threshold tau : gates morphisms. Edge i->j exists iff agent i's
    trust in j (>=) tau. High tau thins the trust graph.
  - cadence K           : gates COMPOSITION depth. Over a horizon H the number
    of broadcast rounds is R = floor(H/K); information (hence morphism
    composition i->k->j) propagates at most R hops. Low K -> many rounds ->
    deep composition; high K -> few rounds -> only short paths.

Invariant per (tau, K) cell, computed on the R-hop reachability closure of the
tau-thinned trust graph:
  - reachability density : fraction of ordered pairs (i!=j) connected;
  - #weakly-connected components (fragmentation);
  - colimit_exists       : 1 iff a single component spans all N agents
                           (a consensus object over the whole collective).

The trust matrix is produced by the system's OWN EMA trust rule
(peer_memory.peer_trust.AsymmetricTrust) under a heterogeneous-reliability
population -- not a hand-set matrix. Deterministic given --seed.

Run: PYTHONPATH=. .venv/bin/python experiments/collective_semantic_memory/phase_diagram_trust_cadence.py
"""
from __future__ import annotations
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

from peer_memory.peer_trust import AsymmetricTrust


def _ring_dist(i, j, N):
    d = abs(i - j)
    return min(d, N - d)


def _channel_reliability(i, j, N):
    """Co-observation reliability of the i<-j channel: neighbours on the
    interaction ring co-observe often (high), distant agents rarely (low).
    This gives trust a TOPOLOGY, not just a per-agent reliability."""
    d = _ring_dist(i, j, N)
    return {0: 1.0, 1: 0.92, 2: 0.72}.get(d, 0.30)


def build_trust_matrix(N, n_updates, seed):
    """T[i][j] = agent i's trust in j from the real EMA rule (AsymmetricTrust),
    driven by ring-topology co-observation reliability."""
    rng = np.random.default_rng(seed)
    T = np.eye(N)
    for i in range(N):
        at = AsymmetricTrust(f"a{i}")
        for _ in range(n_updates):
            for j in range(N):
                if j == i:
                    continue
                p = _channel_reliability(i, j, N)
                at.update_from_outcome(f"a{j}", bool(rng.random() < p))
        for j in range(N):
            T[i, j] = 1.0 if j == i else at.trust_in(f"a{j}")
    return T


def reach_closure(adj, R):
    """Boolean reachability within R hops: (I + adj)^R > 0."""
    N = adj.shape[0]
    base = (adj + np.eye(N)) > 0
    reach = base.copy()
    for _ in range(max(0, R - 1)):
        reach = (reach @ base) > 0
    return reach


def weak_components(reach):
    """#weakly-connected components of the reachability graph (undirected union)."""
    N = reach.shape[0]
    und = reach | reach.T
    seen = [False] * N
    comps = 0
    for s in range(N):
        if seen[s]:
            continue
        comps += 1
        stack = [s]
        while stack:
            u = stack.pop()
            if seen[u]:
                continue
            seen[u] = True
            for v in range(N):
                if und[u, v] and not seen[v]:
                    stack.append(v)
    return comps


def cell(T, tau, R):
    N = T.shape[0]
    adj = (T >= tau).astype(int)
    np.fill_diagonal(adj, 0)
    reach = reach_closure(adj, R)
    off = ~np.eye(N, dtype=bool)
    density = float(reach[off].mean())
    # the colimit SPANS all N (single consensus component) iff every ordered
    # pair composes within R rounds; a disconnected colimit still exists in Pos
    # as a coproduct of sub-consensuses
    colimit = int(np.all(reach[off]))
    return density, weak_components(reach), colimit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--H", type=int, default=64, help="horizon (ticks)")
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--fig", type=str, default=None, help="save heatmap PDF to this path")
    a = ap.parse_args()
    N, H, S = a.N, a.H, a.seeds
    Ts = [build_trust_matrix(N, n_updates=80, seed=s) for s in range(S)]
    T0 = Ts[0]
    print(f"Communication-structure phase diagram  (N={N} agents on an interaction ring, H={H}, {S} seeds)")
    print(f"Trust matrix from real EMA rule (peer_trust.AsymmetricTrust) under ring co-observation.")
    print(f"Trust by ring distance d=1,2,>=3: "
          f"{T0[0,1]:.2f}, {T0[0,2]:.2f}, {T0[0,3]:.2f}  (ring diameter = {N//2})\n")

    TAUS = [0.30, 0.55, 0.80, 0.90]
    KS = [1, 4, 8, 16, 32, 64, 128]  # R = floor(H/K); K=128 > H -> R=0

    def R_of(K):
        return H // K  # broadcast rounds in the horizon

    def avg(tau, K):
        ds, cs = [], []
        for T in Ts:
            d, _, col = cell(T, tau, R_of(K))
            ds.append(d); cs.append(col)
        return float(np.mean(ds)), float(np.mean(cs))

    header = "  tau\\K |" + "".join(f"{K:>7}" for K in KS) + "     (R=" + ",".join(str(R_of(K)) for K in KS) + ")"
    print("Reachability density = fraction of ordered agent pairs whose evidence")
    print("can compose within R = floor(H/K) broadcast rounds (mean over seeds):")
    print(header); print("  " + "-" * 60)
    for tau in TAUS:
        row = f"  {tau:>5.2f} |" + "".join(f"{avg(tau,K)[0]:>7.2f}" for K in KS)
        print(row)

    print("\nP(colimit spans all N) = fraction of seeds with all-to-all")
    print("reachability within R rounds (a consensus object spanning the collective):")
    print(header); print("  " + "-" * 60)
    for tau in TAUS:
        row = f"  {tau:>5.2f} |" + "".join(f"{avg(tau,K)[1]:>7.2f}" for K in KS)
        print(row)

    print("\nReading: low tau -> dense trust graph -> the colimit spans all N at any cadence.")
    print("Raising tau prunes edges to unreliable agents; the collective then fragments")
    print("UNLESS cadence is low enough (R large) for multi-hop paths through still-")
    print("trusted intermediaries to reconnect it. High tau + high K (few rounds) ->")
    print("coproduct of sub-consensuses instead of one spanning colimit. The phase")
    print("boundary is a structural transition, not a tuned number.")

    if a.fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        dens = np.array([[avg(t, K)[0] for K in KS] for t in TAUS])
        coli = np.array([[avg(t, K)[1] for K in KS] for t in TAUS])
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
        for ax, grid, title in [(axes[0], dens, "Reachability density"),
                                (axes[1], coli, r"$P(\mathrm{colimit\ spans\ all\ }N)$")]:
            im = ax.imshow(grid, aspect="auto", cmap="viridis", vmin=0, vmax=1, origin="upper")
            ax.set_xticks(range(len(KS))); ax.set_xticklabels([f"{K}\n(R={R_of(K)})" for K in KS], fontsize=7)
            ax.set_yticks(range(len(TAUS))); ax.set_yticklabels([f"{t:.2f}" for t in TAUS])
            ax.set_xlabel("cadence $K$  (rounds $R=\\lfloor H/K\\rfloor$)"); ax.set_ylabel(r"trust threshold $\tau$")
            ax.set_title(title, fontsize=10)
            for r in range(grid.shape[0]):
                for c in range(grid.shape[1]):
                    ax.text(c, r, f"{grid[r,c]:.2f}", ha="center", va="center",
                            color="white" if grid[r, c] < 0.6 else "black", fontsize=7)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(a.fig, bbox_inches="tight")
        print(f"\nsaved figure -> {a.fig}")


if __name__ == "__main__":
    main()
