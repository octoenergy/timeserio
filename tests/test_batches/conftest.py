import pytest

import dask.distributed


@pytest.fixture(scope='module', autouse=True)
def dask_client():
    client = dask.distributed.Client(processes=False)
    return client
