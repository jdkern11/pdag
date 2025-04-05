from typing import NamedTuple, Callable, ParamSpec, TypeVar, Generic, Any
import networkx as nx
import inspect
import logging

logger = logging.getLogger(__name__)
P = ParamSpec("P")
R = TypeVar("R", bound=Any)


class Node(Generic[R]):
    def __init__(
        self, alias: str, func: Callable[P, R], *args: P.args, **kwargs: P.kwargs
    ):
        _check_signature(func, args, kwargs)
        self.__signature__ = _get_partial_signature(func, *args, **kwargs)
        self.alias: str = alias
        self.func = func
        self.args = args
        arg_names = func.__code__.co_varnames
        if 'self' in arg_names:
            raise ValueError("Not implemented for method calls.")
        for arg, name in zip(args, arg_names):
            kwargs[name] = arg
        self.kwargs = kwargs


    def __call__(self, **kwargs) -> R:
        kwargs = {**self.kwargs, **kwargs}
        return self.func(*[], **kwargs)

    def __repr__(self):
        return self.alias

    def display(self) -> str:
        args = ["{k}={v}" for k, v in self.kwargs.items()]
        rep = ", ".join(args)
        return f"{self.func.__name__}({rep})"


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

        param_input_node = self._get_param_input_node(to_, param)
        if param_input_node is not None:
            if overwrite:
                logger.warning(
                    "Overwriting parameter %s for node %s. The parameter's previous"
                    " input was node %s and is now node %s", param, to_, param_input_node, from_
                )
                if param_input_node.alias != to_.alias:
                    self._remove_param(param_input_node, to_)
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

    def _get_param_input_node(self, node: Node, param: str | None = None) -> Node | None:
        if param is None:
            return None
        # Parameter define in the node kwargs.
        if param in node.kwargs:
            return node
        for edge in self.graph.in_edges(node.alias, data=True):
            if edge[2]["param"] == param:
                return self.nodes[edge[0]]
        return None

    def _remove_param(self, from_: Node, to_: Node):
        del self.graph.edges[from_.alias, to_.alias]['param']

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

    def has_node(self, node: Node):
        a = node.alias
        return a in self.nodes and id(self.nodes[a]) == id(node)

class Status(Generic[R], NamedTuple):
    errored: bool
    output: R | None

class Executor:

    def __init__(self, dag: DAG):
        self.dag = dag
        self.result: dict[str, Status] = {}

    def execute(self) -> dict[str, Status]:
        graph = self.dag.graph
        nodes = self.dag.nodes
        order = nx.topological_sort(graph)
        for alias in order:
            node = nodes[alias]
            self.execute_node(node, save=True)
        return self.result

    def execute_node(self, node: Node, save: bool)-> Status:
        """Executes nodes function based on its input edges

        node: Node within the executor's dag.
        save: If True, saves to executor's result.
        """
        if not self.dag.has_node(node):
            raise ValueError(f"{node} must exist within the excutor's dag.")
        graph = self.dag.graph
        kwargs = node.kwargs.copy()
        edges = graph.in_edges(node.alias, data=True)
        for edge in edges:
            if 'param' in edge[2]:
                kwargs[edge[2]['param']] = self.result[edge[0]].output
        res = node(**kwargs)
        status = Status(errored=False, output=res)
        self.result[node.alias] = status
        return status


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
