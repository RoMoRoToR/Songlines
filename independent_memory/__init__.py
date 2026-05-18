"""Independent memory — fully isolated agents, NO communication.

This is variant (1) from the reviewers' taxonomy: agents that have no
mechanism to exchange information.  Provided as a first-class package
for symmetry with ``distributed_memory/`` (variant 2) and
``peer_memory/`` (variant 3), and as a controlled lower-bound baseline
for ablation studies.

Each ``IndependentAgent`` has its own private event store and concept
graph (built from its own observations only).  Attempting to call
``snapshot()``, ``broadcast()``, or ``receive()`` raises an error.

API surface is intentionally identical in spirit to the other two
runtimes (``spawn_agent``, ``observe``, ``tick``, ``local_query``) so
experiments can switch between the three variants with minimal code.
"""

from independent_memory.independent_agent import IndependentAgent
from independent_memory.independent_runtime import IndependentRuntime

__all__ = ["IndependentAgent", "IndependentRuntime"]
