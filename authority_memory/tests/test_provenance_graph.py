"""Sprint 2 invariant tests --- the provenance DAG must keep
``origin_ids`` and ``provenance_parents`` correctly separated under
relay.  No pytest dependency: run

    PYTHONPATH=. python -m authority_memory.tests.test_provenance_graph

Scope (``docs/FRONTIER_MEMORY_AUTHORITY_2026-08-07/10_CODE_LAYOUT.md``
§5): a synthetic transport chain with no new origin must leave
``n_eff`` unchanged at every hop (the code-level precondition for
Theorem 1, checked here directly --- the full E1 experiment with
several baseline architectures is Sprint 3, not this file).
"""

from __future__ import annotations

from authority_memory.authority_state import AuthorityState
from authority_memory.certificate import Claim
from authority_memory.metrics import n_eff, provenance_amplification_factor
from authority_memory.provenance_graph import (DuplicateOriginError,
                                               ProvenanceGraph)

_checks = []


def check(name, cond):
    _checks.append((name, bool(cond)))


def _claim():
    return Claim(subject="door_X", relation="state",
                object="open", conditions={})


# ── observe() registers exactly one origin -----------------------
def test_observe_creates_single_origin_certificate():
    g = ProvenanceGraph()
    cert, decision = g.observe("cert-1", _claim(), "agent-A", "obs-A-1",
                               world_version=0, observed_at=0)
    check("observe_sets_origin_ids_to_exactly_one_origin",
          cert.origin_ids == {"obs-A-1"})
    check("observe_leaves_provenance_parents_empty",
          cert.provenance_parents == set())
    check("observe_lands_in_quarantined",
          cert.authority_state == AuthorityState.QUARANTINED)
    check("observe_registers_the_origin",
          g.origin("obs-A-1").observing_agent == "agent-A")


def test_duplicate_origin_raises():
    g = ProvenanceGraph()
    g.observe("cert-1", _claim(), "agent-A", "obs-A-1", world_version=0,
             observed_at=0)
    raised = False
    try:
        g.observe("cert-2", _claim(), "agent-A", "obs-A-1",
                 world_version=0, observed_at=1)
    except DuplicateOriginError:
        raised = True
    check("duplicate_origin_id_raises", raised)


# ── Theorem 1, at the code level: a relay chain never grows n_eff --
def test_relay_chain_does_not_amplify_evidence():
    g = ProvenanceGraph()
    cert_a, _ = g.observe("cert-A", _claim(), "agent-A", "obs-A-1",
                         world_version=0, observed_at=0)

    cert_b, _ = g.relay(cert_a, sender="agent-A", receiver="agent-B",
                       new_certificate_id="cert-B", timestamp=1)
    cert_c, _ = g.relay(cert_b, sender="agent-B", receiver="agent-C",
                       new_certificate_id="cert-C", timestamp=2)
    cert_d, _ = g.relay(cert_c, sender="agent-C", receiver="agent-D",
                       new_certificate_id="cert-D", timestamp=3)

    n_eff_by_hop = [n_eff(c) for c in (cert_a, cert_b, cert_c, cert_d)]
    check("n_eff_flat_across_a_pure_relay_chain",
          n_eff_by_hop == [1, 1, 1, 1])
    check("origin_ids_identical_at_every_hop",
          cert_a.origin_ids == cert_b.origin_ids == cert_c.origin_ids
          == cert_d.origin_ids == {"obs-A-1"})
    check("hop_count_grows_by_one_per_relay",
          [g.hop_count(c) for c in (cert_a, cert_b, cert_c, cert_d)]
          == [0, 1, 2, 3])
    check("provenance_parents_accumulates_the_relay_chain",
          cert_b.provenance_parents == {"agent-A"}
          and cert_c.provenance_parents == {"agent-A", "agent-B"}
          and cert_d.provenance_parents
          == {"agent-A", "agent-B", "agent-C"})
    check("relayed_certificates_land_in_quarantined_like_any_receipt",
          all(c.authority_state == AuthorityState.QUARANTINED
             for c in (cert_b, cert_c, cert_d)))


# ── provenance laundering: the cycle A->B->C->D->A is detectable ---
def test_laundering_cycle_detected_but_does_not_amplify():
    g = ProvenanceGraph()
    cert_a, _ = g.observe("cert-A2", _claim(), "agent-A", "obs-A-2",
                         world_version=0, observed_at=0)
    cert_b, _ = g.relay(cert_a, sender="agent-A", receiver="agent-B",
                       new_certificate_id="cert-B2", timestamp=1)
    cert_c, _ = g.relay(cert_b, sender="agent-B", receiver="agent-C",
                       new_certificate_id="cert-C2", timestamp=2)
    cert_d, _ = g.relay(cert_c, sender="agent-C", receiver="agent-D",
                       new_certificate_id="cert-D2", timestamp=3)

    check("laundering_flagged_when_message_returns_to_origin_agent",
          g.is_laundering(cert_d, "agent-A"))
    check("laundering_not_flagged_for_a_genuinely_new_receiver",
          not g.is_laundering(cert_d, "agent-E"))

    cert_a_again, _ = g.relay(cert_d, sender="agent-D",
                              receiver="agent-A",
                              new_certificate_id="cert-A2-returned",
                              timestamp=4)
    check("laundered_return_still_does_not_amplify_n_eff",
          n_eff(cert_a_again) == 1
          and cert_a_again.origin_ids == {"obs-A-2"})


# ── lineage() reconstructs the chain in origin-to-here order -------
def test_lineage_reconstructs_chain_order():
    g = ProvenanceGraph()
    cert_a, _ = g.observe("cert-A3", _claim(), "agent-A", "obs-A-3",
                         world_version=0, observed_at=10)
    cert_b, _ = g.relay(cert_a, sender="agent-A", receiver="agent-B",
                       new_certificate_id="cert-B3", timestamp=11)
    cert_c, _ = g.relay(cert_b, sender="agent-B", receiver="agent-C",
                       new_certificate_id="cert-C3", timestamp=12)

    chain = g.lineage("cert-C3")
    check("lineage_has_three_edges_origin_plus_two_relays",
          len(chain) == 3)
    check("lineage_is_ordered_origin_first",
          chain[0].is_origin and chain[0].certificate_id == "cert-A3"
          and chain[1].certificate_id == "cert-B3"
          and chain[2].certificate_id == "cert-C3")
    check("lineage_edges_carry_correct_sender_receiver",
          chain[1].sender == "agent-A" and chain[1].receiver == "agent-B"
          and chain[2].sender == "agent-B"
          and chain[2].receiver == "agent-C")


# ── independent observations DO grow the aggregate n_eff ------------
def test_independent_observations_grow_aggregate_n_eff():
    g = ProvenanceGraph()
    cert_a, _ = g.observe("cert-A4", _claim(), "agent-A", "obs-A-4",
                         world_version=0, observed_at=0)
    cert_e, _ = g.observe("cert-E4", _claim(), "agent-E", "obs-E-4",
                         world_version=0, observed_at=0)
    cert_f, _ = g.observe("cert-F4", _claim(), "agent-F", "obs-F-4",
                         world_version=0, observed_at=0)

    check("single_certificate_n_eff_is_one",
          n_eff(cert_a) == 1 and n_eff(cert_e) == 1 and n_eff(cert_f) == 1)
    check("group_n_eff_grows_with_each_independent_observation",
          n_eff([cert_a]) == 1
          and n_eff([cert_a, cert_e]) == 2
          and n_eff([cert_a, cert_e, cert_f]) == 3)
    # Relaying cert_a onward must not merge into the independent
    # origins of E/F --- the two phenomena (relay vs. independent
    # observation) must stay distinguishable even when mixed.
    cert_b, _ = g.relay(cert_a, sender="agent-A", receiver="agent-B",
                       new_certificate_id="cert-B4", timestamp=1)
    check("relaying_one_lineage_does_not_borrow_others_origins",
          n_eff([cert_b, cert_e, cert_f]) == 3
          and n_eff([cert_b]) == 1)


# ── PAF is a simple, honestly-guarded ratio -------------------------
def test_provenance_amplification_factor():
    check("paf_flat_authority_gives_one",
          provenance_amplification_factor(0.4, 0.4) == 1.0)
    check("paf_grown_authority_exceeds_one",
          provenance_amplification_factor(0.8, 0.4) == 2.0)
    raised = False
    try:
        provenance_amplification_factor(0.5, 0.0)
    except ValueError:
        raised = True
    check("paf_rejects_zero_origin_authority_instead_of_dividing", raised)


def main():
    for fn in (test_observe_creates_single_origin_certificate,
              test_duplicate_origin_raises,
              test_relay_chain_does_not_amplify_evidence,
              test_laundering_cycle_detected_but_does_not_amplify,
              test_lineage_reconstructs_chain_order,
              test_independent_observations_grow_aggregate_n_eff,
              test_provenance_amplification_factor):
        fn()
    ok = sum(1 for _, c in _checks if c)
    for name, c in _checks:
        print(f"  [{'PASS' if c else 'FAIL'}] {name}")
    print(f"{ok}/{len(_checks)} provenance-graph checks passed")
    return 0 if ok == len(_checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
