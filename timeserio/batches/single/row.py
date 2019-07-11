from typing import Union

from .base import BatchGenerator
from ..utils import ceiling_division


class RowBatchGenerator(BatchGenerator):
    """Generate rows of pd.DataFrame in batches as sub-frames."""

    def __init__(
        self,
        *,
        df,
        batch_size: Union[None, int] = None,
        columns=None
    ):
        self.df = df
        self.batch_size = batch_size
        self.columns = columns

    @property
    def _eff_batch_size(self):
        return self.batch_size or len(self.df)

    def __len__(self):
        return ceiling_division(len(self.df), self._eff_batch_size)

    def __getitem__(self, batch_idx):
        if not len(self):
            raise IndexError('Batch index out of range: Empty batch generator')
        columns = self.columns
        if not columns:
            columns = self.df.columns
        batch_idx = batch_idx % len(self)
        batch_df = self.df[columns].iloc[
            batch_idx * self._eff_batch_size:
            batch_idx * self._eff_batch_size +
            self._eff_batch_size
        ]

        return batch_df
