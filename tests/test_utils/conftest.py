import pytest

import boto3
import moto
import s3fs


@pytest.fixture
def test_bucket_name():
    return 'test_bucket'


@pytest.fixture
def s3(test_bucket_name):
    # writable local S3 system
    with moto.mock_s3():
        client = boto3.client('s3')
        client.create_bucket(Bucket=test_bucket_name)
        yield s3fs.S3FileSystem()
