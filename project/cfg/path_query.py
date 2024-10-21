import networkx as nx

from pyformlang.cfg import CFG, Terminal

from project.cfg.normal_forms import cfg_to_weak_normal_form


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
