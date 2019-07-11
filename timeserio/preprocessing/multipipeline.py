import warnings

from ..pipeline.multipipeline import MultiPipeline  # noqa

MESSAGE = (
    "`timeserio.preprocessing.multipipeline` module is deprecated, "
    "use `timeserio.pipeline` instead."
)
warnings.warn(MESSAGE, DeprecationWarning, stacklevel=2)
