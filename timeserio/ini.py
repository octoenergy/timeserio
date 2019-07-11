"""Defaults and parameters used repeatedly in tests.

Attributes:
    n_jobs (int): default `n_jobs` used in scikit-learn pipelines.
        Set to 1 as parallelism is implemented
        over batches of data when needed.

"""

n_jobs = 1


class Columns:
    """Default column names used in tests.

    Attributes:
        datetime (str): name of the `datetime` column in tests
        id (str): name of the id (categorical) column in tests
        target (str): name of the (real-valued) regression target column

    """

    datetime = 'datetime'
    id = 'id'
    target = 'target'
