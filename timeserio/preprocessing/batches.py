import warnings

from ..batches.single import row as single_row

MESSAGE = (
    "`timeserio.preprocessing.batches` module is deprecated, "
    "use `timeserio.batches` instead."
)
warnings.warn(MESSAGE, DeprecationWarning, stacklevel=2)

PandasBatchGenerator = single_row.RowBatchGenerator
