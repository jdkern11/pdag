from typing import NamedTuple, Callable, ParamSpec, TypeVar, Generic, Any, get_type_hints, get_origin, get_args, Union, Concatenate
from functools import wraps

# Define type variables for the function arguments and return value
P = ParamSpec("P")
R = TypeVar('R')
RetR = TypeVar("RetR")

class Node(Generic[R]):
    def __init__(
        self, func: Callable[P, R], *args: P.args, **kwargs: P.kwargs
    ):
        def inner(*args: P.args, **kwargs: P.kwargs) -> R:
            return func(*args, **kwargs)
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __call__(self) -> R:
        args = [arg() if isinstance(arg, Node) else arg for arg in self.args]
        kwargs = {key: arg() if isinstance(arg, Node) else arg for key, arg in self.kwargs.items()}
        return self.func(*args, **kwargs) # type: ignore


def example_func(x: int, y: str) -> str:
    return f"{x} {y}"

n1 = Node(daggable(example_func), 42, "hello")
daggable_func = daggable(example_func)
n2 = Node(daggable_func, 43, n1, 't')

print(n2())
