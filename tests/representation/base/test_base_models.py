import pytest
import numpy as np

from representation.base.models import TransitionModel


def test_transition_model_is_abstract():
    with pytest.raises(TypeError):
        TransitionModel("t", "g", "o")
