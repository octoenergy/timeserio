import pytest

import boto3
import moto


@pytest.fixture
def test_bucket_name():
    return 'test_bucket'


@pytest.fixture
def s3(test_bucket_name):
    # writable local S3 system
    with moto.mock_s3():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=test_bucket_name)
        yield
