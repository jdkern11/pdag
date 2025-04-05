import functools
import logging
import pytest
import pdag


def add(x: int, y: int) -> int:
    return x + y


def test_create_node():
    n1 = pdag.Node("n1", add, 1, y=2)
    res = n1()
    assert res == 3
    n2 = pdag.Node("n2", add, x=n1, y=3)

    n2 = pdag.Node("n2", functools.partial(add, 1, 2))
    res = n2()
    assert res == 3

    with pytest.raises(TypeError):
        pdag.Node("err", add, 1, y=2, z=4) # type: ignore


def test_remove_edge(caplog):
    caplog.set_level(logging.INFO)
    n1 = pdag.Node("n1", lambda: 1)
    n2 = pdag.Node("n2", add, y=3)
    n3 = pdag.Node("n3", add, y=3)
    dag = pdag.DAG()
    dag.remove_edge(n1, n2)
    assert "Can't remove edge as it doesn't exist" in caplog.text
    caplog.clear()

    dag.add_edge(n1, n2, "x")
    assert len(dag.nodes) == 2
    assert dag.graph.number_of_edges() == 1
    dag.remove_edge(n1, n2)
    assert len(dag.nodes) == 0
    assert dag.graph.number_of_edges() == 0

    dag.add_edge(n1, n2, "x")
    dag.add_edge(n2, n3, "x")
    # Can't remove non-leaf node.
    with pytest.raises(ValueError):
        dag.remove_edge(n1, n2)


def test_add_edge(caplog):
    caplog.set_level(logging.INFO)
    n1 = pdag.Node("n1", lambda: 1)
    n2 = pdag.Node("n2", add, y=3)
    dag = pdag.DAG()
    dag.add_edge(n1, n2, "x")
    assert "Edge already added" not in caplog.text
    dag.add_edge(n1, n2, "x")
    assert "Edge already added" in caplog.text


def test_edge_creates_cycle():
    n1 = pdag.Node("n1", add, y=3)
    n2 = pdag.Node("n2", add, y=3)
    n3 = pdag.Node("n3", add, y=3)
    dag = pdag.DAG()
    dag.add_edge(n1, n2, "x")
    with pytest.raises(ValueError):
        dag.add_edge(n2, n1, "x")
    dag.add_edge(n2, n3, "x")
    with pytest.raises(ValueError):
        dag.add_edge(n3, n1, "x")


def test_override(caplog):
    n1 = pdag.Node("n1", lambda: 1)
    n2 = pdag.Node("n2", add, y=3)
    dag = pdag.DAG()
    with pytest.raises(ValueError):
        dag.add_edge(n1, n2, "y")
    dag.add_edge(n1, n2, "x")
    n3 = pdag.Node("n3", add, y=3)
    with pytest.raises(ValueError):
        dag.add_edge(n3, n2, "x")
    dag.add_edge(n3, n2, "x", overwrite=True)
    assert "Overwriting parameter x" in caplog.text
    for edge in dag.graph.in_edges(n2.alias, data=True):
        if edge[0] == n1.alias:
            assert len(edge[2]) == 0


def test_param_is_valid():
    def func1(x):
        return x

    assert pdag._param_is_valid(func1, "x")
    assert not pdag._param_is_valid(func1, "y")


def test_execute():
    n1 = pdag.Node("1", add, 1, y=1)
    n2 = pdag.Node("2", add, y=2)
    n3 = pdag.Node("3", add)
    dag = pdag.DAG()
    dag.add_edge(n1, n2, "x")
    dag.add_edge(n2, n3, "x")
    dag.add_edge(n1, n3, "y")
    executor = pdag.Executor(dag)
    outputs = executor.execute()
    assert outputs["1"].output == 2
    assert outputs["2"].output == 4
    assert outputs["3"].output == 6

    n1 = pdag.Node("1", add, 1, y=1)
    n2 = pdag.Node("2", add, y=2)
    dag = pdag.DAG()
    dag.add_edge(n1, n2, "z")
