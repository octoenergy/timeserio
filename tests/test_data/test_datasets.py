from timeserio.data import datasets


def test_load_iris_df():
    df = datasets.load_iris_df()

    assert "species" in df
    assert "sepal_width_cm" in df
    assert len(df)
