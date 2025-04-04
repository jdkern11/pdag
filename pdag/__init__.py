from typing import NamedTuple, Callable, ParamSpec, TypeVar, Generic
import networkx as nx
import inspect
import logging

logger = logging.getLogger(__name__)
P = ParamSpec("P")
R = TypeVar("R")


class Node(Generic[R]):
    def __init__(
        self, alias: str, func: Callable[P, R], *args: P.args, **kwargs: P.kwargs
    ):
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
        args = tuple(f"{arg}" for arg in self.args) + tuple(
            "{k}={v}" for k, v in self.kwargs.items()
        )
        rep = ", ".join(args)
        return f"{self.func.__name__}({rep})"


class Status(Generic[R], NamedTuple):
    errored: bool
    output: R | None


class DAG:
    def __init__(self):
        self.graph: nx.DiGraph[str] = nx.DiGraph()
        self.nodes: dict[str, Node] = {}

    def add_edge(
        self, from_: Node, to_: Node, param: str | None = None, overwrite: bool = False
    ):
        """Add edge from node `from_` to node `to_`, filling param.

        Args:
            from_: Parent node.
            to_: Child node.
            param: Input parameter for the `to_` node.
            overwrite: If True, will overwrite param if already input to the
                to_ node.

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

        if param is not None and not _param_is_valid(to_.func, param):
            raise ValueError(f"{param} is an invalid parameter for {to_.func.__name__}")

        if self._has_param_input(to_, param):
            if overwrite:
                logger.warning("Overwriting parameter %s for node %s", param, to_)
            else:
                raise ValueError(
                    f"{param} is already added to the `to_` node. Pass `overwrite=True`"
                    " if you're sure you wan't to overwrite it."
                )

        if self._edge_creates_cycle(from_, to_):
            raise ValueError("This creates a cycle.")

        self._add_edge_unsafe(from_, to_, param)

    def _validate_alias(self, node: Node):
        """Raises value error if node alias would overwrite another node"""
        a = node.alias
        if a in self.nodes and id(self.nodes[a]) != id(node):
            raise ValueError(
                f"Alias of {node.display()} overwrites node "
                f"{self.nodes[a].display()} in the graph"
            )

    def remove_edge(self, from_: Node, to_: Node):
        if not self._edge_exists(from_, to_):
            logger.info("Can't remove edge as it doesn't exist.")
            return
        if not self._is_leaf_node(to_):
            raise ValueError("Can't remove non-leaf node.")
        self.graph.remove_edge(from_.alias, to_.alias)
        self._remove_node_if_disjoint(from_)
        self._remove_node_if_disjoint(to_)

    def _is_leaf_node(self, node: Node) -> bool:
        return self.graph.out_degree(node.alias) == 0

    def _remove_node_if_disjoint(self, node: Node):
        if (
            self.graph.in_degree(node.alias) == 0
            and self.graph.out_degree(node.alias) == 0
        ):
            self.graph.remove_node(node.alias)
            del self.nodes[node.alias]

    def _edge_exists(self, from_: Node, to_: Node) -> bool:
        return self.graph.has_edge(from_.alias, to_.alias)

    def _has_param_input(self, node: Node, param: str | None = None) -> bool:
        if param is None:
            return False
        if param in node.kwargs:
            return True
        # returns from_, to_, parameters.
        for edge in self.graph.in_edges(node.alias, data=True):
            if edge[2]["param"] == param:
                return True
        return False

    def _edge_creates_cycle(self, from_: Node, to_: Node) -> bool:
        if to_.alias not in self.nodes or from_.alias not in self.nodes:
            return False
        return to_.alias in nx.ancestors(self.graph, from_.alias)

    def _add_edge_unsafe(self, from_: Node, to_: Node, param: str | None = None):
        """If cycle and param checks aren't done, this is dangerous because it could
        result in a cycle or invalid input parameter.."""
        self.graph.add_edge(from_.alias, to_.alias, param=param)
        self.nodes[from_.alias] = from_
        self.nodes[to_.alias] = to_


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
        param
        for param in signature.parameters.values()
        if param.name not in bound_signature.arguments
    ]
    return inspect.Signature(remaining_params)
