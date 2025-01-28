import networkx as nx

from pyformlang.cfg import CFG
from pyformlang.rsa import RecursiveAutomaton
from scipy.sparse import csr_matrix

from project.cfg.normal_forms import cfg_to_weak_normal_form
from project.cfg.rsm import rsm_to_nfa
from project.regular.to_automaton import graph_to_nfa
from project.regular.automatons import AdjacencyMatrixFA, intersect_automata


def hellings_based_cfpq(
    cfg: CFG,
    graph: nx.DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    graph_nodes = set(graph.nodes)
    start_nodes = start_nodes if start_nodes else graph_nodes
    final_nodes = final_nodes if final_nodes else graph_nodes

    cfg = cfg_to_weak_normal_form(cfg)

    r = set((N, v, v) for v in graph_nodes for N in cfg.get_nullable_symbols())

    for v, u, label in graph.edges.data("label"):
        for prod in cfg.productions:
            if (
                len(prod.body) == 1
                and prod.body[0] in cfg.terminals
                and prod.body[0].value == label
            ):
                r.add((prod.head, v, u))

    while True:
        new_triples = set()

        for N_i, v, u in r:
            for N_j, v1, u1 in r:
                if u == v1:
                    for prod in cfg.productions:
                        if (
                            len(prod.body) == 2
                            and prod.body[0] == N_i
                            and prod.body[1] == N_j
                            and (prod.head, v, u1) not in r
                        ):
                            new_triples.add((prod.head, v, u1))

        if not new_triples:
            break
        r = r.union(new_triples)

    return set(
        (v, u)
        for sym, v, u in r
        if sym == cfg.start_symbol and v in start_nodes and u in final_nodes
    )


def matrix_based_cfpq(
    cfg: CFG,
    graph: nx.DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    graph_nodes = set(graph.nodes)
    start_nodes = start_nodes if start_nodes else graph_nodes
    final_nodes = final_nodes if final_nodes else graph_nodes

    node_to_idx_mapping = {node: i for i, node in enumerate(graph_nodes)}
    idx_to_node_mapping = {i: node for i, node in enumerate(graph_nodes)}

    n = len(graph_nodes)
    cfg = cfg_to_weak_normal_form(cfg)
    decomposition = {var: csr_matrix((n, n), dtype=bool) for var in cfg.variables}

    for u, v, label in graph.edges.data("label"):
        for prod in cfg.productions:
            if (
                len(prod.body) == 1
                and prod.body[0] in cfg.terminals
                and prod.body[0].value == label
            ):
                decomposition[prod.head][
                    node_to_idx_mapping[u], node_to_idx_mapping[v]
                ] = True

    for N in cfg.get_nullable_symbols():
        decomposition[N].setdiag(True)

    changed = True
    while changed:
        changed = False
        for prod in cfg.productions:
            if len(prod.body) != 2:
                continue
            A_i = prod.head
            A_j, A_k = prod.body
            head_matrix = decomposition[A_i] + (decomposition[A_j] @ decomposition[A_k])
            if (decomposition[A_i] != head_matrix).nnz != 0:
                changed = True
                decomposition[A_i] = head_matrix

    return {
        (idx_to_node_mapping[u], idx_to_node_mapping[v])
        for u, v in zip(*decomposition[cfg.start_symbol].nonzero())
        if idx_to_node_mapping[u] in start_nodes
        and idx_to_node_mapping[v] in final_nodes
    }


def tensor_based_cfpq(
    rsm: RecursiveAutomaton,
    graph: nx.DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    rsm_automaton = AdjacencyMatrixFA(rsm_to_nfa(rsm))
    graph_automaton = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))

    for sym in rsm.labels:
        if sym not in graph_automaton.adj_decomposition.keys():
            graph_automaton.adj_decomposition[sym] = csr_matrix(
                (graph_automaton.number_of_states, graph_automaton.number_of_states),
                dtype=bool,
            )

    changed = True

    while changed:
        changed = False
        intersection = intersect_automata(rsm_automaton, graph_automaton)
        transitive_closure = intersection.transitive_сlosure()
        idx_to_state_mapping = {v: k for k, v in intersection.states.items()}

        for row_id, col_id in zip(*transitive_closure.nonzero()):
            (row_symbol, row_rsm_state), row_graph_state = idx_to_state_mapping[row_id]
            (col_symbol, col_rsm_state), col_graph_state = idx_to_state_mapping[col_id]

            dfa = rsm.boxes[row_symbol].dfa
            if (
                row_symbol == col_symbol
                and row_rsm_state in dfa.start_states
                and col_rsm_state in dfa.final_states
            ):
                if not graph_automaton.adj_decomposition[row_symbol][
                    graph_automaton.states[row_graph_state],
                    graph_automaton.states[col_graph_state],
                ]:
                    graph_automaton.adj_decomposition[row_symbol][
                        graph_automaton.states[row_graph_state],
                        graph_automaton.states[col_graph_state],
                    ] = True
                    changed = True

    idx_to_node_mapping = {v: k for k, v in graph_automaton.states.items()}
    return {
        (idx_to_node_mapping[u], idx_to_node_mapping[v])
        for u, v in zip(*graph_automaton.adj_decomposition[rsm.initial_label].nonzero())
        if idx_to_node_mapping[u] in start_nodes
        and idx_to_node_mapping[v] in final_nodes
    }
