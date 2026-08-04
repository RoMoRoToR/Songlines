"""Songlines core — frame-free landmark alignment and route recovery.

``song_target`` re-derives the correspondence between a song's
landmark constellations and the receiver's own observations at
consumption time (nothing coordinate-valued is stored), then
dead-reckons over the beat chain to a target the receiver has never
seen.  Landmark-less songs cannot anchor; beat-less songs cannot
reach the unseen --- both fail closed by construction.  Safety modes
(anchor consensus, unimodal clustering, loop closure) live here.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from experiments.warp.semantic_identity import cosine

GridXY = Tuple[float, float]


def song_target(song, band_fps, sim: float, min_anchors: int = 1,
                closure_tol: float = 0.0,
                simfn=None, unimodal_tol: float = 0.0,
                return_support: bool = False):
    """Frame-free consumption: mutually-unique landmark matching over
    the receiver's own observations + dead reckoning over beats.

    Safety mode (min_anchors >= 2): dead reckoning is allowed only
    from a CONSENSUS of anchors whose pairwise displacements agree
    with the beat chain (the loop-closure rule of the identity layer,
    which single-anchor consumption used to bypass --- the N1
    lesson)."""
    fsim = simfn or cosine
    cs = [c for c in song[:-1] if c.get("sig")]
    matches: List[Tuple[int, GridXY]] = []
    for j, c in enumerate(cs):
        scored = [(xy, fsim(c["sig"], s))
                  for xy, s in band_fps.items() if s]
        cands = [(xy, v) for xy, v in scored if v >= sim]
        if not cands:
            continue
        if unimodal_tol > 0:
            # continuous anchoring: dense observation points around
            # one place legitimately co-match; under aliasing take
            # the BEST cluster and let consensus loop closure
            # adjudicate --- a wrong-cluster anchor cannot agree with
            # the beat chain against the other anchors
            clusters: List[List[Tuple[GridXY, float]]] = []
            for xy, v in sorted(cands, key=lambda p: -p[1]):
                for cl in clusters:
                    if (abs(cl[0][0][0] - xy[0])
                            + abs(cl[0][0][1] - xy[1])) <= unimodal_tol:
                        cl.append((xy, v))
                        break
                else:
                    clusters.append([(xy, v)])
            best_cl = max(clusters, key=lambda cl: max(v for _, v in cl))
            # sub-grid refinement: similarity-weighted centroid of the
            # winning cluster (anchor quantization was the entire
            # near-miss tail of the fail-open distribution)
            wts = [max(v - sim + 1e-6, 1e-6) for _, v in best_cl]
            ax = sum(xy[0] * w for (xy, _), w in zip(best_cl, wts)) \
                / sum(wts)
            ay = sum(xy[1] * w for (xy, _), w in zip(best_cl, wts)) \
                / sum(wts)
            matches.append((j, (ax, ay)))
        else:
            if len(cands) != 1:
                continue
            xy = cands[0][0]
            back = [k for k, c2 in enumerate(cs)
                    if c2.get("sig")
                    and fsim(band_fps[xy], c2["sig"]) >= sim]
            if back == [j]:
                matches.append((j, xy))
    if len(matches) < max(1, min_anchors):
        return None
    # loop closure: displacement between matched anchors must agree
    # with the beat-chain sum (within tolerance, for continuous
    # substrates)
    for (j1, p1), (j2, p2) in zip(matches, matches[1:]):
        seg = [song[k].get("beat") for k in range(j1 + 1, j2 + 1)]
        if any(b is None for b in seg):
            return None
        bx = sum(b[0] for b in seg)
        by = sum(b[1] for b in seg)
        if (abs((p2[0] - p1[0]) - bx) > closure_tol
                or abs((p2[1] - p1[1]) - by) > closure_tol):
            return None
    j_last, p_last = matches[-1]
    tail = [song[k].get("beat") for k in range(j_last + 1, len(song))]
    if any(b is None for b in tail):
        return None
    dx = sum(b[0] for b in tail)
    dy = sum(b[1] for b in tail)
    tgt = (p_last[0] + dx, p_last[1] + dy)
    return (tgt, len(matches)) if return_support else tgt
