import networkx as nx

from itertools import product
from scipy.sparse import spmatrix
from pyformlang.finite_automaton import Symbol

from regular.automatons import AdjacencyMatrixFA, intersect_automata, sparse_types
from regular.to_automaton import regex_to_dfa, graph_to_nfa


def tensor_based_rpq(
    regex: str,
    graph: nx.MultiDiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
    sparse_format="csr",
) -> set[tuple[int, int]]:
    regex_automaton = AdjacencyMatrixFA(regex_to_dfa(regex), sparse_format)
    graph_automaton = AdjacencyMatrixFA(
        graph_to_nfa(graph, start_nodes, final_nodes), sparse_format
    )
    intersection = intersect_automata(regex_automaton, graph_automaton, sparse_format)
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


def init_front(
    regex_automaton: AdjacencyMatrixFA,
    graph_automaton: AdjacencyMatrixFA,
    sparse_type,
) -> spmatrix:
    front = sparse_type(
        (
            regex_automaton.number_of_states * len(graph_automaton.start_states),
            graph_automaton.number_of_states,
        )
    )

    graph_start_states = sorted(graph_automaton.start_states)
    for i, graph_state_name in enumerate(graph_start_states):
        for regex_state_name in regex_automaton.start_states:
            regex_idx = (
                regex_automaton.states[regex_state_name]
                + i * regex_automaton.number_of_states
            )
            graph_idx = graph_automaton.states[graph_state_name]
            front[regex_idx, graph_idx] = True

    return front


def update_front(
    front: spmatrix,
    regex_automaton: AdjacencyMatrixFA,
    graph_automaton: AdjacencyMatrixFA,
    symbols: set[Symbol],
) -> spmatrix:
    next_front = front
    regex_n_of_states = regex_automaton.number_of_states

    for sym in symbols:
        sym_front = front @ graph_automaton.adj_decomposition[sym]
        for i in range(len(graph_automaton.start_states)):
            sym_front[i * regex_n_of_states : (i + 1) * regex_n_of_states] = (
                regex_automaton.adj_decomposition[sym].T
                @ sym_front[i * regex_n_of_states : (i + 1) * regex_n_of_states]
            )
        next_front += sym_front

    return next_front


def ms_bfs_based_rpq(
    regex: str,
    graph: nx.MultiDiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
    sparse_format="csr",
) -> set[tuple[int, int]]:
    regex_automaton = AdjacencyMatrixFA(regex_to_dfa(regex), sparse_format)
    graph_automaton = AdjacencyMatrixFA(
        graph_to_nfa(graph, start_nodes, final_nodes), sparse_format
    )

    symbols = regex_automaton.symbols.intersection(graph_automaton.symbols)

    sparse_type = sparse_types[sparse_format]
    front = init_front(regex_automaton, graph_automaton, sparse_type)
    visited = front
    while front.count_nonzero() > 0:
        new_front = update_front(front, regex_automaton, graph_automaton, symbols)
        front = new_front > visited
        visited += front

    pairs = set()
    regex_n_of_states = regex_automaton.number_of_states
    graph_start_states = sorted(graph_automaton.start_states)
    graph_states_mapping = {v: k for k, v in graph_automaton.states.items()}

    for i, graph_start_name in enumerate(graph_start_states):
        visited_block = visited[i * regex_n_of_states : (i + 1) * regex_n_of_states]
        for regex_final_name in regex_automaton.final_states:
            for reached in visited_block.getrow(
                regex_automaton.states[regex_final_name]
            ).indices:
                if graph_states_mapping[reached] in graph_automaton.final_states:
                    pairs.add((graph_start_name, graph_states_mapping[reached]))

    return pairs
