from typing import NamedTuple, Callable, ParamSpec, TypeVar, Generic
import inspect
import logging
from collections import defaultdict, deque

logger = logging.getLogger(__name__)
P = ParamSpec("P")
R = TypeVar("R")


class Node(Generic[R]):
    def __init__(self, func: Callable[P, R], *args: P.args, **kwargs: P.kwargs):
        _check_signature(func, args, kwargs)
        self.func = func
        self.alias: str | None = None
        self.args = args
        self.kwargs = kwargs

    def execute(self, **kwargs) -> R:
        kwargs = {**self.kwargs, **kwargs}
        return self.func(*self.args, **kwargs)

    def __repr__(self):
        return self.alias or self.__str__()

    def __str__(self):
        rep = ", ".join(self.args + tuple((f"{k}={v}" for k, v in self.kwargs.items())))
        return f"{self.func.__name__}({rep})"


class Status(Generic[R], NamedTuple):
    errored: bool
    output: R | None


class DAG:
    def __init__(self):
        self.output_edges = defaultdict(set)
        self.input_edges = defaultdict(set)
        self.node_inputs = defaultdict(dict)
        self.node_outputs = defaultdict(Status)

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

        if param in self.node_inputs[to_]:
            logger.warning(f"%s already in %s. Overwriting.", param, to_)
        self.node_inputs[to_][param] = from_

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
            self.input_edges.pop(node, None)
            self.output_edges.pop(node, None)
            self.node_inputs.pop(node, None)

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

        self.node_inputs = defaultdict(dict)
        self.node_outputs = defaultdict(Status)

    def execute(self) -> dict[Node, Status]:
        in_degree = defaultdict(int)
        for node in self.input_edges:
            in_degree[node] = len(self.input_edges[node])

        queue = deque([node for node, degree in in_degree.items() if degree == 0])
        while queue:
            node = queue.popleft()
            kwargs = {}
            for param, ancestor in self.node_inputs[node].items():
                kwargs[param] = self.node_outputs[ancestor].output
            res = node.execute(**kwargs)
            self.node_outputs[node] = Status(errored=False, output=res)
            for dependent in self.output_edges[node]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        return self.node_outputs


def _param_is_valid(func: Callable, param: str) -> bool:
    signature = inspect.signature(func)
    return param in signature.parameters


def _check_signature(func: Callable, args, kwargs):
    signature = inspect.signature(func)
    try:
        signature.bind_partial(*args, **kwargs)
    except TypeError as e:
        raise TypeError(f"Invalid arguments for {func.__name__}: {e}")
