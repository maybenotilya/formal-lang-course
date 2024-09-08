import cfpq_data
import networkx as nx

from dataclasses import dataclass
from typing import Set, Tuple
from pathlib import Path


@dataclass(eq=True)
class GraphInfo:
    number_of_nodes: int
    number_of_edges: int
    edges_labels: Set[str]


def load_graph(graph_name: str) -> nx.MultiDiGraph:
    graph_path = cfpq_data.download(graph_name)
    graph = cfpq_data.graph_from_csv(graph_path)
    return graph


def get_graph_info(graph: nx.MultiDiGraph):
    return GraphInfo(
        number_of_nodes=graph.number_of_nodes(),
        number_of_edges=graph.number_of_edges(),
        edges_labels={label for _, _, label in graph.edges.data("label")},
    )


def labeled_two_cycles_graph_to_dot(
    n: int, m: int, labels: Tuple[str, str], save_path: Path
):
    graph = cfpq_data.labeled_two_cycles_graph(n, m, labels=labels)
    nx.nx_pydot.write_dot(graph, save_path)
