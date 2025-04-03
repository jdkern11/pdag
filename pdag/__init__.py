from typing import NamedTuple, Callable, ParamSpec, TypeVar
import inspect
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)
P = ParamSpec('P')
R = TypeVar('R')

class Node:
    def __init__(self, func: Callable[P, R], node_alias: str | None = None, *args: P.args, **kwargs: P.kwargs):
        self.func = func
        self.alias = node_alias
        self.args = args
        self.kwargs = kwargs
        self._executed = False
        self._result: R | None = None

    def execute(self) -> R:
        if self._executed:
            return self._result
        self._result = self.func(*self.args, **self.kwargs)
        self._executed = True
        return self._result

    def __repr__(self):
        rep = ", ".join(self.args + tuple((f"{k}={v}" for k, v in self.kwargs.items())))
        return f"{self.func.__name__}({rep})"

    def __str__(self):
        return self.alias or self.__repr__()


class DAG:
    def __init__(self):
        self.output_edges = defaultdict(set)
        self.input_edges = defaultdict(set)

    def __repr__(self):
        return str({str(k): [str(n) for n in v] for k, v in self.output_edges.items()})

    def add_edge(self, from_: Node, to_: Node, param: str | None = None):
        if self._edge_exists(from_, to_):
            logger.info("Edge already added.")
            return

        if self._edge_creates_cycle(from_, to_):
            self.remove_edge(from_, to_)
            raise ValueError("This creates a cycle.")

        if param is not None and not _param_is_valid(to_.func, param):
            raise ValueError(f"{param} is an invalid parameter for {to_.func.__name__}")

        self._add_edge_unsafe(from_, to_, param)

    def _edge_exists(self, from_: Node, to_: Node) -> bool:
        return to_ in self.output_edges[from_]

    def _add_edge_unsafe(self, from_: Node, to_: Node, param: str | None = None):
        """If cycle and param checks aren't done, this is dangerous to do."""
        self.output_edges[from_].add(to_)
        self.input_edges[to_].add(from_)
        if to_ not in self.output_edges:
            self.output_edges[to_] = set()
        if from_ not in self.input_edges:
            self.input_edges[from_] = set()

        # TODO figure out param mapping. Shouldn't modify Node at all.

    def remove_edge(self, from_: Node, to_: Node):
        if not self._edge_exists(from_, to_):
            logger.info("Can't remove edge as it doesn't exist.")
            return

        if not self._is_leaf_node(to_):
            raise ValueError("Can't remove a non-leaf node.")

        self.output_edges[from_].remove(to_)
        self.input_edges[to_].remove(from_)
        self._remove_node_if_disconnected(to_)
        self._remove_node_if_disconnected(from_)

    def _remove_node_if_disconnected(self, node: Node):
        if self._disconnected_node(node):
            self.input_edges.pop(node)
            self.output_edges.pop(node)

    def _disconnected_node(self, node: Node) -> bool:
        return len(self.input_edges[node]) == 0 and len(self.output_edges[node]) == 0

    def _is_leaf_node(self, node: Node):
        return len(self.output_edges[node]) == 0

    def _edge_creates_cycle(self, from_: Node, to_: Node) -> bool:
        visited = set()

        def dfs(node: Node) -> bool:
            visited.add(node)
            for ancestor in self.input_edges[node]:
                if ancestor == to_:
                    return True
                if ancestor not in visited:
                    if dfs(ancestor):
                        return True
            return False

        return dfs(from_)


def _param_is_valid(func: Callable, param: str) -> bool:
    signature = inspect.signature(func)
    return param in signature.parameters
