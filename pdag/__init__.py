from typing import NamedTuple
import inspect
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class Node:
    def __init__(self, func: callable, *args, node_alias: str | None = None, **kwargs):
        self.func = func
        self.alias = node_alias
        self.args = args
        self.kwargs = kwargs
        self._executed = False
        self._result = None

    def execute(self):
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
        self.node_input_edges = defaultdict(set)

    def __repr__(self):
        return str({str(k): [str(n) for n in v] for k, v in self.node_input_edges.items()})

    def add_edge(self, from_: Node, to_: Node, param: str | None = None):
        if self._edge_exists(from_, to_):
            logger.info("Edge already added.")
            return

        if self._edge_creates_cycle(from_, to_):
            raise ValueError("This creates a cycle.")

        if not _param_is_valid(to_.func, param):
            raise ValueError(
                f"{param} is an invalid parameter for {to_.func.__name__}"
            )

        self._add_edge_unsafe(from_, to_, param)


    def _edge_exists(self, from_: Node, to_: Node) -> bool:
        return from_ in self.node_input_edges[to_]

    def _add_edge_unsafe(self, from_: Node, to_: Node, param: str | None = None):
        """If cycle and param checks aren't done, this is dangerous to do."""
        self.node_input_edges[to_].add(from_)
        if from_ not in self.node_input_edges:
            self.node_input_edges[from_] = set()
        if param is not None:
            to_.kwargs[param] = from_._result

    def _dfs(self, root: Node):
        stack = [root]
        visited = set()
        while stack:
            node = stack.pop()
            yield node
                
    def _edge_creates_cycle(self, from_: Node, to_: Node) -> bool:
        # Should only occur with initial inserts of nodes and immediately stop.
        if from_ not in self.node_input_edges or to_ not in self.node_input_edges:
            return False
        if to_ in self.node_input_edges[from_]:
            return True
        for ancestor in self.node_input_edges[from_]:
            if self._edge_creates_cycle(ancestor, to_):
                return True

def _param_is_valid(func: callable, param: str) -> bool:
    signature = inspect.signature(func)
    return param in signature.parameters
