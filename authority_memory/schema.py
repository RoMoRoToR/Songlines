"""Authority Memory core — relational schemas (Sprint 9 slice: the
representation and structural-isomorphism matcher; induction from
labelled examples is ``schema_induction.py``).

Content-similarity matching (embedding distance, LCS/NW graph
alignment such as G1, ``experiments/song_grammar/exp_g1_graph_analogy.py``)
transfers a structure only as far as SURFACE features carry over
between worlds --- and G1 specifically has no notion of abstract role
or causal effect at all (B10, ``docs/CLAIM_EVIDENCE_MATRIX.md``: "no
roles/causality... only structural").  A ``RelationalSchema`` instead
represents a pattern as abstract ROLE SLOTS connected by typed
RELATIONS, with declared PRECONDITIONS/EFFECTS --- matching it against
a new world means finding an assignment of roles to that world's OWN
nodes such that the required relation-typed edges exist, using only
the new world's own observed connectivity, never any cross-world tag
or embedding comparison.  This is why it can transfer with ZERO
surface overlap between worlds: it never depended on surface overlap
in the first place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class RelationalSchema:
    """An abstract pattern: ``roles`` are slot names (never bound to
    any particular world's content), ``relations`` are
    (role_a, relation_type, role_b) triples that must hold between
    whichever nodes eventually fill those roles, ``preconditions``/
    ``effects`` are the applicability/expected-outcome the pattern
    carries once instantiated, and ``exception_ids`` names known
    counterexamples (structural exceptions are Sprint 9's follow-on,
    E8; the field exists now so a schema object never needs to change
    shape to carry them later)."""
    schema_id: str
    roles: Tuple[str, ...]
    relations: Tuple[Tuple[str, str, str], ...]
    preconditions: Dict[str, Any] = field(default_factory=dict)
    effects: Dict[str, Any] = field(default_factory=dict)
    exception_ids: Tuple[str, ...] = ()


@dataclass(frozen=True)
class WorldGraph:
    """One concrete world's observed local structure: ``node_tags``
    is surface content (differs freely between worlds, may share
    NOTHING with any other world); ``edges`` is relation-typed
    connectivity (node_id_a, relation_type, node_id_b) --- the ONLY
    thing ``match_schema_to_graph`` ever looks at.  Node ids carry no
    role information themselves.
    """
    world_id: str
    node_tags: Dict[str, str]
    edges: Tuple[Tuple[str, str, str], ...]


def match_schema_to_graph(schema: RelationalSchema, graph: WorldGraph
                          ) -> List[Dict[str, str]]:
    """Every assignment of ``schema.roles`` to distinct nodes of
    ``graph`` under which every required relation triple corresponds
    to an actual edge --- pure relation-type isomorphism search, never
    consulting ``graph.node_tags``.  Returns ALL such bindings (there
    may be more than one, or none); the caller decides how to use
    that --- ``unique_match`` below is the fail-closed policy the rest
    of this series already uses for identity matching (W7-W10):
    commit only when the match is unambiguous.
    """
    node_ids = list(graph.node_tags)
    if len(node_ids) < len(schema.roles):
        return []
    edge_set = set(graph.edges)
    matches = []
    for perm in permutations(node_ids, len(schema.roles)):
        binding = dict(zip(schema.roles, perm))
        if all((binding[a], rel, binding[b]) in edge_set
              for a, rel, b in schema.relations):
            matches.append(binding)
    return matches


def unique_match(schema: RelationalSchema,
                 graph: WorldGraph) -> Optional[Dict[str, str]]:
    """The single binding, if ``match_schema_to_graph`` found exactly
    one --- ``None`` on zero OR more than one match. Ambiguity is
    refused, not resolved by guessing (fail-closed)."""
    matches = match_schema_to_graph(schema, graph)
    return matches[0] if len(matches) == 1 else None


def schema_bits(schema: RelationalSchema, *, role_bits: int = 8,
                relation_type_bits: int = 4,
                precondition_effect_bits: int = 16) -> int:
    """A registered, explicit bit codec for a ``RelationalSchema`` ---
    the same discipline as the runtime's song codec
    (``songlines/record.py``: one codec, applied everywhere, so a
    bits comparison across architectures is honest, not a post-hoc
    number).  A relation triple costs two role references plus a
    relation-type id; roles/preconditions/effects are a small fixed
    overhead paid ONCE, not per world the schema is later matched
    against.
    """
    roles_cost = len(schema.roles) * role_bits
    relations_cost = len(schema.relations) * (2 * role_bits
                                              + relation_type_bits)
    return roles_cost + relations_cost + precondition_effect_bits
