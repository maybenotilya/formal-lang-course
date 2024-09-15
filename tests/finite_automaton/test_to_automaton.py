import pytest

from typing import Set

from project.utils import load_graph, get_graph_info
from project.finite_automaton.to_automaton import regex_to_dfa, graph_to_nfa


@pytest.mark.parametrize(
    "regex, expected_symbols",
    [
        ("formal-lang-course", set(["formal-lang-course"])),
        ("(z|q)(x|w)(c|e)", set("zxcqwe")),
        ("g(o*)l", set("gol")),
    ],
)
def test_regex_to_dfa_symbols(regex: str, expected_symbols: Set[str]):
    dfa = regex_to_dfa(regex)

    assert dfa.symbols == expected_symbols


@pytest.mark.parametrize("graph_name", ["generations"])
@pytest.mark.parametrize("start_states", [set(), set(range(5))])
@pytest.mark.parametrize("final_states", [set(), set(range(5))])
def test_graph_to_nfa_info(
    graph_name: str, start_states: Set[int], final_states: Set[int]
):
    graph = load_graph(graph_name)
    graph_info = get_graph_info(graph)
    nfa = graph_to_nfa(graph, start_states, final_states)

    assert nfa.start_states == start_states if start_states else nfa.states
    assert nfa.final_states == final_states if final_states else nfa.states
    assert len(nfa.states) == graph_info.number_of_nodes
    assert nfa.get_number_transitions() == graph_info.number_of_edges
    assert nfa.symbols == graph_info.edges_labels
