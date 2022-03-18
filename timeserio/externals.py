import builtins
import logging
from types import ModuleType


__all__ = [
    "keras", "HABEMUS_KERAS", "tensorflow", "HABEMUS_TENSORS",
    "tensorpandas", "HABEMUS_TENSOR_EXT"
]


logger = logging.getLogger(__name__)


class NotFoundModule(ModuleType):

    def __init__(self, name, *args, **kwargs):
        logger.warn(
            "Module `%s` could not be imported. Creating mock instead.",
            name
        )
        super().__init__(name, *args, **kwargs)

    def __getattr__(self, attr):
        # Doctest attempts to unwrap everything to see if it can doctest it
        # so we need to raise an AttributeError in that case rather than the
        # specific ModuleNotFoundError
        if attr == "__wrapped__":
            raise AttributeError(
                "'NotFoundModule' has no attribute '__wrapped__'"
            )
        raise ModuleNotFoundError(
            f"Can not access `{self.__name__}.{attr}`, "
            f"module `{self.__name__}` was not found during import."
        )

    def __str__(self):
        return f'<mock module "{self.__name__}">'

    def __repr__(self):
        return f'NotFoundModule("{self.__name__}")'


def optional_import(module_name):
    """Import optional module by name.

    Returns a mock object if not found.
    """
    try:
        module = builtins.__import__(module_name)
        success = True
    except ModuleNotFoundError:
        module = NotFoundModule(module_name)
        success = False

    return module, success


tensorflow, HABEMUS_TENSORS = optional_import("tensorflow")
if HABEMUS_TENSORS:
    keras, HABEMUS_KERAS = tensorflow.keras, True
else:
    keras, HABEMUS_KERAS = NotFoundModule("keras"), False

tensorpandas, HABEMUS_TENSOR_EXT = optional_import("tensorpandas")
