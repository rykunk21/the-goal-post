import pytest

from representation.base.imputer import ImputationBuilder


def test_imputation_builder_is_abstract():
    with pytest.raises(TypeError):
        ImputationBuilder()
