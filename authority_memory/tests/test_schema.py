"""Sprint 9 invariant tests --- relational schema matching must be
purely structural (indifferent to tag content), fail-closed on
ambiguity, and induction must reject examples that do not actually
exhibit the claimed pattern.  No pytest dependency: run

    PYTHONPATH=. python -m authority_memory.tests.test_schema
"""

from __future__ import annotations

from authority_memory.schema import (RelationalSchema, WorldGraph,
                                     match_schema_to_graph, schema_bits,
                                     unique_match)
from authority_memory.schema_induction import induce_schema

_checks = []


def check(name, cond):
    _checks.append((name, bool(cond)))


ROUTE_SCHEMA = RelationalSchema(
    schema_id="route", roles=("bottleneck", "passage", "resource"),
    relations=(("bottleneck", "leads_to", "passage"),
              ("passage", "leads_to", "resource")))


def _world(world_id, b_tag="x", p_tag="y", r_tag="z", d_tag="w"):
    return WorldGraph(
        world_id=world_id,
        node_tags={"n1": b_tag, "n2": p_tag, "n3": r_tag, "n4": d_tag},
        edges=(("n1", "leads_to", "n2"), ("n2", "leads_to", "n3"),
              ("n4", "alt_leads_to", "n3")))


# ── matching is purely structural, indifferent to tag content -------
def test_match_ignores_tag_content_entirely():
    w1 = _world("w1", "red_door", "hallway", "water", "decoy1")
    w2 = _world("w2", "totally_unrelated_string_47", "xyz123",
               "qqqqqqq", "nope")
    m1 = unique_match(ROUTE_SCHEMA, w1)
    m2 = unique_match(ROUTE_SCHEMA, w2)
    check("match_succeeds_regardless_of_tag_vocabulary",
          m1 == {"bottleneck": "n1", "passage": "n2", "resource": "n3"}
          and m2 == {"bottleneck": "n1", "passage": "n2", "resource": "n3"})


def test_match_fails_closed_when_pattern_absent():
    no_pattern = WorldGraph(
        world_id="no-pattern", node_tags={"n1": "a", "n2": "b", "n3": "c"},
        edges=(("n1", "unrelated", "n2"),))
    check("no_match_when_relations_absent",
          match_schema_to_graph(ROUTE_SCHEMA, no_pattern) == [])
    check("unique_match_none_when_no_pattern",
          unique_match(ROUTE_SCHEMA, no_pattern) is None)


def test_match_refuses_on_ambiguity():
    # Two disjoint occurrences of the SAME relation-type chain ---
    # more than one valid binding, so the fail-closed policy must
    # refuse rather than pick one arbitrarily.
    ambiguous = WorldGraph(
        world_id="ambiguous",
        node_tags={"n1": "a", "n2": "b", "n3": "c",
                  "n4": "d", "n5": "e", "n6": "f"},
        edges=(("n1", "leads_to", "n2"), ("n2", "leads_to", "n3"),
              ("n4", "leads_to", "n5"), ("n5", "leads_to", "n6")))
    matches = match_schema_to_graph(ROUTE_SCHEMA, ambiguous)
    check("ambiguous_world_has_multiple_matches", len(matches) > 1)
    check("unique_match_refuses_on_ambiguity",
          unique_match(ROUTE_SCHEMA, ambiguous) is None)


def test_match_needs_enough_nodes():
    tiny = WorldGraph(world_id="tiny", node_tags={"n1": "a", "n2": "b"},
                      edges=(("n1", "leads_to", "n2"),))
    check("no_match_when_fewer_nodes_than_roles",
          match_schema_to_graph(ROUTE_SCHEMA, tiny) == [])


# ── schema_bits: fixed, amortised cost -------------------------------
def test_schema_bits_is_fixed_and_positive():
    bits = schema_bits(ROUTE_SCHEMA)
    check("schema_bits_positive", bits > 0)
    check("schema_bits_matches_hand_computation",
          bits == 3 * 8 + 2 * (2 * 8 + 4) + 16)


# ── induce_schema: agreement required, bad examples rejected ---------
def test_induce_schema_from_agreeing_examples():
    examples = [
        (_world("A", "red_door", "hallway", "water"),
        {"bottleneck": "n1", "passage": "n2", "resource": "n3"}),
        (_world("B", "blue_gate", "tunnel", "charger"),
        {"bottleneck": "n1", "passage": "n2", "resource": "n3"}),
        (_world("C", "rock_opening", "canyon", "food"),
        {"bottleneck": "n1", "passage": "n2", "resource": "n3"}),
    ]
    schema = induce_schema(examples, "route",
                           ("bottleneck", "passage", "resource"))
    check("induced_schema_has_expected_relations",
          schema.relations == ROUTE_SCHEMA.relations)
    # The induced schema must itself transfer to a brand-new world.
    novel = _world("novel", "aaaaaaaaaa", "bbbbbbbbbb", "cccccccccc")
    check("induced_schema_transfers_to_a_novel_world",
          unique_match(schema, novel)
          == {"bottleneck": "n1", "passage": "n2", "resource": "n3"})


def test_induce_schema_rejects_example_missing_the_pattern():
    bad_example = WorldGraph(
        world_id="bad", node_tags={"n1": "a", "n2": "b", "n3": "c"},
        edges=(("n1", "unrelated", "n2"),))
    examples = [
        (_world("A"), {"bottleneck": "n1", "passage": "n2",
                      "resource": "n3"}),
        (bad_example, {"bottleneck": "n1", "passage": "n2",
                      "resource": "n3"}),
    ]
    raised = False
    try:
        induce_schema(examples, "route",
                     ("bottleneck", "passage", "resource"))
    except ValueError:
        raised = True
    check("induce_schema_raises_on_non_conforming_example", raised)


def test_induce_schema_requires_at_least_two_examples():
    raised = False
    try:
        induce_schema([(_world("only-one"),
                       {"bottleneck": "n1", "passage": "n2",
                       "resource": "n3"})],
                     "route", ("bottleneck", "passage", "resource"))
    except ValueError:
        raised = True
    check("induce_schema_raises_on_single_example", raised)


def main():
    for fn in (test_match_ignores_tag_content_entirely,
              test_match_fails_closed_when_pattern_absent,
              test_match_refuses_on_ambiguity,
              test_match_needs_enough_nodes,
              test_schema_bits_is_fixed_and_positive,
              test_induce_schema_from_agreeing_examples,
              test_induce_schema_rejects_example_missing_the_pattern,
              test_induce_schema_requires_at_least_two_examples):
        fn()
    ok = sum(1 for _, c in _checks if c)
    for name, c in _checks:
        print(f"  [{'PASS' if c else 'FAIL'}] {name}")
    print(f"{ok}/{len(_checks)} schema checks passed")
    return 0 if ok == len(_checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
