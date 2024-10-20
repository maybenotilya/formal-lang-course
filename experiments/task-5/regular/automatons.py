import numpy as np

from pyformlang.finite_automaton import NondeterministicFiniteAutomaton, Symbol
from scipy.sparse import (
    csr_matrix,
    bsr_matrix,
    csc_matrix,
    coo_matrix,
    dia_matrix,
    dok_matrix,
    lil_matrix,
    kron,
)
from typing import Iterable
from itertools import product

sparse_types = {
    "csr": csr_matrix,
    "bsr": bsr_matrix,
    "csc": csc_matrix,
    "coo": coo_matrix,
    "dia": dia_matrix,
    "dok": dok_matrix,
    "lil": lil_matrix,
}


class AdjacencyMatrixFA:
    def __init__(
        self, automaton: NondeterministicFiniteAutomaton = None, sparse_format="csr"
    ):
        if automaton is None:
            self.number_of_states = 0
            self.states = dict()
            self.start_states = set()
            self.final_states = set()
            self.adj_decomposition = dict()
            return

        sparse_type = sparse_types[sparse_format]

        graph = automaton.to_networkx()
        self.states = {state_name: i for i, state_name in enumerate(graph.nodes)}
        self.number_of_states = len(self.states)

        self.start_states = set(
            state_name
            for state_name in self.states.keys()
            if state_name in automaton.start_states
        )

        self.final_states = set(
            state_name
            for state_name in self.states.keys()
            if state_name in automaton.final_states
        )

        transitions = {
            sym: np.zeros((self.number_of_states, self.number_of_states), dtype=bool)
            for sym in automaton.symbols
        }

        for st1, st2, label in graph.edges(data="label"):
            if not label:
                continue

            from_state = self.states[st1]
            to_state = self.states[st2]
            transitions[label][from_state, to_state] = 1

        self.adj_decomposition = {
            label: sparse_type(matrix) for label, matrix in transitions.items()
        }

    @property
    def symbols(self):
        return set(self.adj_decomposition.keys())

    def accepts(self, word: Iterable[Symbol]) -> bool:
        current_states = {state for state in self.start_states}

        for sym in word:
            next_states = set()

            if self.adj_decomposition.get(sym) is None:
                return False

            for from_state in current_states:
                for to_state in self.states.keys():
                    if self.adj_decomposition[sym][
                        self.states[from_state], self.states[to_state]
                    ]:
                        next_states.add(to_state)

            current_states = next_states

        for state in current_states:
            if state in self.final_states:
                return True

        return False

    def transitive_сlosure(self) -> np.ndarray:
        A = np.eye(self.number_of_states, dtype=bool)

        for adj_matrix in self.adj_decomposition.values():
            A |= adj_matrix.toarray()

        closure = A.copy()
        for _ in range(self.number_of_states):
            prev = closure.copy()
            closure = np.matmul(closure, A).astype(bool)
            if np.array_equal(closure, prev):
                return closure

        return closure

    def is_empty(self) -> bool:
        closure = self.transitive_сlosure()
        for start_state in self.start_states:
            for final_state in self.final_states:
                if closure[self.states[start_state], self.states[final_state]]:
                    return False

        return True


def intersect_automata(
    automaton1: AdjacencyMatrixFA, automaton2: AdjacencyMatrixFA, sparse_format="csr"
) -> AdjacencyMatrixFA:
    intersection = AdjacencyMatrixFA()

    intersection.number_of_states = (
        automaton1.number_of_states * automaton2.number_of_states
    )
    intersection.states = {
        (st1, st2): automaton1.states[st1] * automaton2.number_of_states
        + automaton2.states[st2]
        for st1, st2 in product(automaton1.states.keys(), automaton2.states.keys())
    }

    intersection.start_states = set(
        state_name
        for state_name in intersection.states.keys()
        if state_name[0] in automaton1.start_states
        and state_name[1] in automaton2.start_states
    )

    intersection.final_states = set(
        state_name
        for state_name in intersection.states.keys()
        if state_name[0] in automaton1.final_states
        and state_name[1] in automaton2.final_states
    )

    for key in automaton1.adj_decomposition.keys():
        if automaton2.adj_decomposition.get(key) is None:
            continue
        intersection.adj_decomposition[key] = kron(
            automaton1.adj_decomposition[key],
            automaton2.adj_decomposition[key],
            format=sparse_format,
        )

    return intersection
