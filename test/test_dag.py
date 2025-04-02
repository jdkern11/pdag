import functools
import logging
import pytest
import pdag


def add(x, y):
    return x + y


def test_create_node():
    n1 = pdag.Node(add, 1, y=2)
    res = n1.execute()
    assert res == 3
    assert n1._executed
    assert n1._result == 3

    n2 = pdag.Node(functools.partial(add, 1, 2))
    res = n2.execute()
    assert res == 3
    assert n2._executed
    assert n2._result == 3


def test_remove_edge(caplog):
    caplog.set_level(logging.INFO)
    n1 = pdag.Node(lambda: 1)
    n2 = pdag.Node(add, y=3)
    n3 = pdag.Node(add, y=3)
    dag = pdag.DAG()
    dag.remove_edge(n1, n2)
    assert "Can't remove edge as it doesn't exist" in caplog.text
    caplog.clear()

    dag.add_edge(n1, n2, "x")
    assert len(dag.input_edges) == 2
    assert len(dag.output_edges) == 2
    dag.remove_edge(n1, n2)
    assert len(dag.input_edges) == 0
    assert len(dag.output_edges) == 0

    dag.add_edge(n1, n2, "x")
    dag.add_edge(n2, n3, "x")
    # Can't remove non-leaf node.
    with pytest.raises(ValueError):
        dag.remove_edge(n1, n2)


def test_add_edge(caplog):
    caplog.set_level(logging.INFO)
    n1 = pdag.Node(lambda: 1)
    n2 = pdag.Node(add, y=3)
    dag = pdag.DAG()
    dag.add_edge(n1, n2, "x")
    assert "Edge already added" not in caplog.text
    dag.add_edge(n1, n2, "x")
    assert "Edge already added" in caplog.text


def test_edge_creates_cycle():
    n1 = pdag.Node(add, y=3, node_alias="n1")
    n2 = pdag.Node(add, y=3, node_alias="n2")
    n3 = pdag.Node(add, y=3, node_alias="n3")
    dag = pdag.DAG()
    dag.add_edge(n1, n2, "x")
    with pytest.raises(ValueError):
        dag.add_edge(n2, n1, "x")
    dag.add_edge(n2, n3, "x")
    with pytest.raises(ValueError):
        dag.add_edge(n3, n1, "x")
    print(dag)


def test_param_is_valid():
    def func1(x):
        return x

    assert pdag._param_is_valid(func1, "x")
    assert not pdag._param_is_valid(func1, "y")
