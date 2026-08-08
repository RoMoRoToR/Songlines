"""Authority Memory core — schema induction from labelled examples
(Sprint 9 slice).

Verifies that several heterogeneous, labelled groundings of the same
candidate pattern actually agree on its relational structure, and
packages the agreed structure as one ``RelationalSchema``.  This is
NOT open-ended pattern discovery from raw unlabelled experience (that
remains future work --- ``07_ROADMAP_SPRINTS.md`` §"later": open-ended
relational schemas is scoped beyond E7/E8).  A single grounding could
never distinguish "this is a general pattern" from "this happened to
work once here"; requiring agreement across multiple, surface-
distinct examples before committing is the whole of what makes this
an induction step rather than a memorised instance.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

from authority_memory.schema import RelationalSchema, WorldGraph


def induce_schema(examples: Sequence[Tuple[WorldGraph, Dict[str, str]]],
                  schema_id: str, roles: Sequence[str],
                  relation_type: str = "leads_to",
                  preconditions: Dict[str, object] = None,
                  effects: Dict[str, object] = None) -> RelationalSchema:
    """``examples`` are (world_graph, role_binding) pairs, each
    asserting that this world exhibits the SAME chain of roles
    connected by ``relation_type`` (roles[i] -> roles[i+1] for
    consecutive roles).  Raises if any example's own graph does not
    actually contain the required edge for its own labelled binding
    --- induction fails loudly on a bad example rather than silently
    inducing a schema that does not even fit its own training data.
    """
    if len(examples) < 2:
        raise ValueError("induce_schema needs at least two examples to "
                        "distinguish a pattern from a coincidence")
    relations = tuple((roles[i], relation_type, roles[i + 1])
                      for i in range(len(roles) - 1))
    for graph, binding in examples:
        edge_set = set(graph.edges)
        missing = [(a, rel, b) for a, rel, b in relations
                  if (binding.get(a), rel, binding.get(b)) not in edge_set]
        if missing:
            raise ValueError(
                f"example {graph.world_id!r} does not exhibit the "
                f"required relation(s) {missing} under its own labelled "
                f"binding {binding}")
    return RelationalSchema(schema_id=schema_id, roles=tuple(roles),
                            relations=relations,
                            preconditions=dict(preconditions or {}),
                            effects=dict(effects or {}))
