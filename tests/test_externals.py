import pytest

from timeserio.externals import optional_import


def test_absent_imports():
    """Test that mock module import is created."""
    module, HABEMUS_MODULE = optional_import("not_real_module")

    assert not HABEMUS_MODULE
    assert module.__name__ == "not_real_module"
    with pytest.raises(ModuleNotFoundError):
        _ = module.layers
