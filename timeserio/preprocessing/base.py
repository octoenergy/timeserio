import warnings

from ..pipeline.pipeline import *  # noqa

MESSAGE = (
    "`timeserio.preprocessing.base` module is deprecated, "
    "use `timeserio.pipeline` instead."
)
warnings.warn(MESSAGE, DeprecationWarning, stacklevel=2)
