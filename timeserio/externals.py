import builtins
import logging
from types import ModuleType


__all__ = ["keras", "HABEMUS_KERAS", "tensorflow", "HABEMUS_TENSORS"]


logger = logging.getLogger(__name__)


class NotFoundModule(ModuleType):

    def __init__(self, name, *args, **kwargs):
        logger.warn(
            "Module `%s` could not be imported. Creating mock instead.",
            name
        )
        super().__init__(name, *args, **kwargs)

    def __getattr__(self, attr):
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


keras, HABEMUS_KERAS = optional_import("keras")
tensorflow, HABEMUS_TENSORS = optional_import("tensorflow")
