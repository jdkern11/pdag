from __future__ import annotations
from collections import defaultdict
from typing import NamedTuple, Callable, ParamSpec, TypeVar, Generic, Any, get_type_hints, get_origin, get_args, Union, Concatenate, cast
import uuid
from contextlib import contextmanager
from functools import wraps

# Define type variables for the function arguments and return value
P = ParamSpec("P")
R = TypeVar('R')

class Node(Generic[R]):
    def __init__(
        self, func: Callable[P, R], *args: P.args, **kwargs: P.kwargs
    ):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.id = uuid.uuid4()

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return self.func(*self.args, **self.kwargs)

    def __repr__(self) -> str:
        return str(self.id)

    def __str__(self) -> str:
        rep = f"{self.func.__name__}("
        rep += ", ".join([str(x) for x in self.args])
        rep += ", ".join([f"{k}={str(v)}" for k, v in self.kwargs.items()])
        rep += ")"
        return rep

class Success(Generic[R]):
    def __init__(self, dag: DAG, node: Node, res: R):
        self.dag = dag
        self._res = res
        self.node = node

    @property
    def res(self) -> R:
        self.dag.add_input(self.node)
        return self._res

class Failure:
    def __init__(self, dag: DAG, node: Node, error_msg: str, error_type: type):
        self.dag = dag
        self.error_msg = error_msg
        self.error_type = error_type
        self.node = node

class DAG:
    def __init__(self, parent_dag: DAG | None = None):
        self.parent_dag = parent_dag
        self.graph: dict[Node, list[Node]] = defaultdict(list)
        self.execution_order: list[Node] = []
        self.result: dict[Node, Success | Failure] = {}
        self._input_nodes: list[Node] = []

    def add_input(self, node):
        self._input_nodes.append(node)

    def add_node(self, func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> Success[R] | Failure:
        n = Node(func, *args, **kwargs)
        self.execution_order.append(n)
        self.add_edges(n)
        try:
            res = n()
            return Success(dag, n, res)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            return Failure(self, n, str(e), type(e))

    def add_edges(self, node: Node):
        while self._input_nodes:
            input_node = self._input_nodes.pop()
            self.graph[input_node].append(node)

    def print_execution_order(self):
        for node in self.execution_order:
            print(node)

    def print_graph(self):
        print(self.graph)

def example_func(x: int, y: str) -> str:
    return f"{x} {y}"

dag = DAG()
r1 = dag.add_node(example_func, 42, "hello")
if isinstance(r1, Failure):
    raise ValueError(f"Node 1 failed: {r1.error_type}, {r1.error_msg}")
r2 = dag.add_node(example_func, 43, r1.res)

print(r2)
dag.print_execution_order()
dag.print_graph()
