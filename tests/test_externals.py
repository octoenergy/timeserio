import pytest

from timeserio.externals import optional_import


def test_absent_imports(mocker):
    """Test that mock keras import is created."""
    _import = mocker.patch("timeserio.externals.builtins.__import__")
    _import.side_effect = ModuleNotFoundError()

    keras, HABEMUS_KERAS = optional_import("keras")

    assert not HABEMUS_KERAS
    assert keras.__name__ == "keras"
    with pytest.raises(ModuleNotFoundError):
        _ = keras.layers
