import networkx as nx

from itertools import product

from project.finite_automaton.automatons import AdjacencyMatrixFA, intersect_automata
from project.finite_automaton.to_automaton import regex_to_dfa, graph_to_nfa


def tensor_based_rpq(
    regex: str,
    graph: nx.MultiDiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    regex_automaton = AdjacencyMatrixFA(regex_to_dfa(regex))
    graph_automaton = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))
    intersection = intersect_automata(regex_automaton, graph_automaton)
    closure = intersection.transitive_сlosure()

    return set(
        (graph_start, graph_final)
        for graph_start, graph_final, regex_start, regex_final in product(
            graph_automaton.start_states,
            graph_automaton.final_states,
            regex_automaton.start_states,
            regex_automaton.final_states,
        )
        if closure[
            intersection.states[(regex_start, graph_start)],
            intersection.states[(regex_final, graph_final)],
        ]
    )
