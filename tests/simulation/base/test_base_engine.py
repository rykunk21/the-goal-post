import pytest

from simulation.base.engine import Simulator


def test_simulator_is_abstract():
    with pytest.raises(TypeError):
        Simulator()
