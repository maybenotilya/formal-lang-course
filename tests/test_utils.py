import pytest
import networkx as nx

from typing import Tuple
from pathlib import Path

from project.utils import (
    GraphInfo,
    get_graph_info_by_name,
    labeled_two_cycles_graph_to_dot,
)


@pytest.mark.parametrize(
    ["graph_name", "expected_graph_info"],
    [
        (
            "wc",
            GraphInfo(
                number_of_nodes=332, number_of_edges=269, edges_labels={"a", "d"}
            ),
        ),
        (
            "generations",
            GraphInfo(
                number_of_nodes=129,
                number_of_edges=273,
                edges_labels={
                    "type",
                    "rest",
                    "first",
                    "onProperty",
                    "intersectionOf",
                    "equivalentClass",
                    "someValuesFrom",
                    "hasValue",
                    "hasSex",
                    "inverseOf",
                    "sameAs",
                    "hasParent",
                    "hasChild",
                    "range",
                    "hasSibling",
                    "versionInfo",
                    "oneOf",
                },
            ),
        ),
    ],
)
def test_get_graph_info_by_name(graph_name: str, expected_graph_info: GraphInfo):
    graph_info = get_graph_info_by_name(graph_name)
    assert graph_info == expected_graph_info


@pytest.mark.parametrize(
    ["n", "m", "labels", "expected_graph_path"],
    [
        (3, 5, ("a", "b"), Path("tests/test_data/utils/expected_graph_3_5.dot")),
        (
            6,
            11,
            ("first awesome label", "second amazing label"),
            Path("tests/test_data/utils/expected_graph_6_11.dot"),
        ),
    ],
)
def test_labeled_two_cycles_graph_to_dot(
    n: int, m: int, labels: Tuple[str, str], expected_graph_path: Path
):
    save_path = Path("actual_graph.dot")
    labeled_two_cycles_graph_to_dot(n, m, labels, save_path)

    expected_graph = nx.nx_pydot.read_dot(expected_graph_path)
    actual_graph = nx.nx_pydot.read_dot(save_path)

    assert nx.utils.graphs_equal(expected_graph, actual_graph)
