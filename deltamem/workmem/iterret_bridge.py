"""Bridge: run ITERRET's real retrieve/reflect/route loop, stop BEFORE
answer_node, and return accumulated_evidence -- a list[str] directly
compatible with deltamem.workmem.osam_workmem.populate_osam_from_evidence.

Nothing in iterret/*.py is modified. This file only calls existing,
unmodified ITERRET functions in the order graph.py already wires them
(retrieve -> reflect -> route -> [retrieve again | stop]), just without
ever calling answer_node.
"""
from __future__ import annotations

from typing import List

from iterret.ctc_graph import CueTagContentGraph
from iterret.experience_bank import ExperienceBank
from iterret.llm_client import LLMClient
from iterret.nodes import reflect_node, retrieve_node, route_after_reflect
from iterret.state import DEFAULT_MAX_ITERATIONS, new_state


def get_iterret_evidence(
    question: str,
    graph: CueTagContentGraph,
    bank: ExperienceBank,
    llm: LLMClient,
    *,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> List[str]:
    """Run retrieve/reflect/route for up to max_iterations rounds.
    Returns state["accumulated_evidence"] WITHOUT ever calling answer_node.
    """
    state = new_state(question, max_iterations=max_iterations)
    for _ in range(max_iterations):
        state = retrieve_node(state, graph, bank, llm)
        state = reflect_node(state, graph, bank, llm)
        if route_after_reflect(state) == "answer":
            break
    return list(state.get("accumulated_evidence", []))
