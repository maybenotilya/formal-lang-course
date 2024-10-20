from pyformlang.finite_automaton import (
    DeterministicFiniteAutomaton,
    NondeterministicFiniteAutomaton,
    EpsilonNFA,
)
from pyformlang.regular_expression import Regex
from networkx import MultiDiGraph
from typing import Set


def regex_to_dfa(regex: str) -> DeterministicFiniteAutomaton:
    enfa = Regex(regex).to_epsilon_nfa()
    assert isinstance(enfa, EpsilonNFA)

    return enfa.remove_epsilon_transitions().to_deterministic().minimize()


def graph_to_nfa(
    graph: MultiDiGraph, start_states: Set[int] = None, final_states: Set[int] = None
) -> NondeterministicFiniteAutomaton:
    enfa = NondeterministicFiniteAutomaton.from_networkx(graph)

    start_states = start_states if start_states else set(graph)
    final_states = final_states if final_states else set(graph)

    for state in start_states:
        enfa.add_start_state(state)

    for state in final_states:
        enfa.add_final_state(state)

    return enfa.remove_epsilon_transitions()
