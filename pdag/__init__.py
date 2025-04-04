from typing import NamedTuple, Callable, ParamSpec, TypeVar, Generic
import networkx
import inspect
import logging
from collections import defaultdict, deque

logger = logging.getLogger(__name__)
P = ParamSpec("P")
R = TypeVar("R")


class Node(Generic[R]):
    def __init__(self, alias: str, func: Callable[P, R], *args: P.args, **kwargs: P.kwargs):
        _check_signature(func, args, kwargs)
        self.alias: str = alias
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.__signature__ = _get_partial_signature(func, *args, **kwargs)

    def __call__(self, **kwargs) -> R:
        kwargs = {**self.kwargs, **kwargs}
        return self.func(*self.args, **kwargs)

    def __repr__(self):
        return self.alias

    def display(self) -> str:
        rep = ", ".join(self.args + tuple((f"{k}={v}" for k, v in self.kwargs.items())))
        return f"{self.func.__name__}({rep})"


class Status(Generic[R], NamedTuple):
    errored: bool
    output: R | None


class DAG:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.nodes: dict[str, Node] = {}

    def __repr__(self):
        return str({str(k): [str(n) for n in v] for k, v in self.output_edges.items()})

    def add_edge(self, from_: Node, to_: Node, param: str | None = None):
        """Add edge from node `from_` to node `to_`, filling param.

        Args:
            from_: Parent node.
            to_: Child node.
            param: Input parameter for the `to_` node.

        Raises:
            ValueError if a node's alias is the same as anothers in the graph,
            if adding an edge creates a cycle, or if an input parameter is invalid
            for the `to_` node.
        """
        self._validate_alias(from_)
        self._validate_alias(to_)
        if self._edge_exists(from_, to_):
            logger.info("Edge already added.")
            return

        if self._edge_creates_cycle(from_, to_):
            raise ValueError("This creates a cycle.")

        if param is not None and not _param_is_valid(to_.func, param):
            raise ValueError(f"{param} is an invalid parameter for {to_.func.__name__}")

        self._add_edge_unsafe(from_, to_, param)

    def _validate_alias(self, node: Node):
        """Raises value error if node alias would overwrite another node"""
        a = node.alias
        if a in self.nodes and id(self.nodes[a]) != id(node):
            raise ValueError(f"Alias of {node.display()} overwrites node {self.nodes[a].display()} in the graph")


    def _edge_exists(self, from_: Node, to_: Node) -> bool:
        return self.graph.has_edge(from_, to_)

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
        if self.to_ not in self.nodes:
            return False
        return to_ in nx.ancestors(self.graph, from_)

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
            res = node(**kwargs)
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


def _get_partial_signature(func, *args, **kwargs) -> inspect.Signature:
        signature = inspect.signature(func)
        bound_signature = signature.bind_partial(*args, **kwargs)
        bound_signature.apply_defaults()
        remaining_params = [
            param for param in signature.parameters.values()
            if param.name not in bound_signature.arguments
        ]
        return inspect.Signature(remaining_params)

