import abc
from bisect import bisect_right

import numpy as np
import pandas as pd

from ..single.base import BatchGenerator
from ..utils import ceiling_division


class ChunkedBatchGenerator(BatchGenerator):
    """Base class for chunked/partitioned batch generators."""

    batch_aggregator = 1

    @abc.abstractmethod
    def __init__(self, batch_size=None, **kwargs):
        self.batch_size = batch_size
        self.__subgen_lengths = None

    @abc.abstractproperty
    def chunks(self):
        """Return an iterable of chunks.

        A chunk is passed to `self.make_subgen`
        """
        raise NotImplementedError

    @abc.abstractmethod
    def make_subgen(self, chunk) -> BatchGenerator:
        """Return a sub-generator for a single chunk.

        """
        raise NotImplementedError

    def _subgen_length(self, chunk):
        """Optimization for slow subgens."""
        if self.batch_size is None:
            return 1
        else:
            return len(self.make_subgen(chunk))

    @property
    def subgen_lengths(self):
        """List of lengths of individual sub-generators."""
        if self.__subgen_lengths is None:
            self.__subgen_lengths = [
                self._subgen_length(chunk)
                for chunk in self.chunks
            ]
        return self.__subgen_lengths

    @property
    def subgen_index_bounds(self):
        """Return List of subgen index bounds.

        First value is 0
        Last value is self.__len__()
        """
        return np.cumsum([0] + self.subgen_lengths)

    @property
    def num_subbatches(self):
        return sum(self.subgen_lengths)

    def __len__(self):
        return ceiling_division(self.num_subbatches, self.batch_aggregator)

    def find_subbatch_in_subgens(self, subbatch_idx):
        if not (0 <= subbatch_idx < self.num_subbatches):
            raise IndexError('Batch index out of range')
        subgen_idx = bisect_right(self.subgen_index_bounds, subbatch_idx) - 1
        idx_in_subgen = subbatch_idx - self.subgen_index_bounds[subgen_idx]
        return subgen_idx, idx_in_subgen

    def __getitem__(self, batch_idx):
        if not len(self):
            raise IndexError('Batch index out of range: Empty batch generator')
        batch_idx = batch_idx % len(self)
        subbatch_idx_start = batch_idx * self.batch_aggregator
        subbatch_idx_end = min(
            (batch_idx + 1) * self.batch_aggregator,
            self.num_subbatches
        )
        batch = pd.DataFrame()
        for subbatch_idx in range(subbatch_idx_start, subbatch_idx_end):
            subgen_idx, idx_in_subgen = self.find_subbatch_in_subgens(
                subbatch_idx
            )
            chunk = self.chunks[subgen_idx]
            subgen = self.make_subgen(chunk)
            subbatch = subgen[idx_in_subgen]
            batch = batch.append(subbatch, ignore_index=True)
        return batch
