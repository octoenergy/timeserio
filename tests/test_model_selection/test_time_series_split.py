import numpy as np
import pandas as pd
import pytest

from timeserio.model_selection import PandasTimeSeriesSplit


def input_df(teams, index_offset=0, num_days=12):
    days = pd.date_range("2019-01-01", periods=num_days).tolist()
    df = pd.DataFrame(
        {
            "team": sum([[team] * num_days for team in teams], []),
            "val": np.concatenate(
                [np.random.random(num_days) for _ in range(len(teams))]
            ),
            "date": days * len(teams),
        }
    )
    df.index += index_offset
    return df


@pytest.mark.parametrize(
    "df",
    [
        input_df(["A"]),
        input_df(["A", "B"]),
        input_df(["A", "B", "C"]),
        input_df(["A"], index_offset=1),
        input_df(["A", "B"], index_offset=1),
    ],
)
def test_pandas_time_series_split(df):
    splitter = PandasTimeSeriesSplit(
        groupby="team", datetime_col="date", n_splits=3
    )

    count = 0
    for X_idx, y_idx in splitter.split(df):
        count += 1
        X, y = df.iloc[X_idx], df.iloc[y_idx]

        for team in df.team.unique():
            X_team, y_team = X[X.team == team], y[y.team == team]
            min_train, max_train = X_team["date"].min(), X_team["date"].max()
            min_test, max_test = y_team["date"].min(), y_team["date"].max()

            assert (min_test - max_train).days == 1
            assert (max_train - min_train).days == count * 3 - 1
            assert (max_test - min_test).days == 3 - 1

    assert count == 3


def test_raises_nonunique():
    df = input_df(teams=["A"])
    df.index = [1] * len(df)

    splitter = PandasTimeSeriesSplit(
        groupby="team", datetime_col="date", n_splits=3
    )

    with pytest.raises(ValueError):
        for X_idx, y_idx in splitter.split(df):
            pass


def test_raises_non_ascending():
    df = input_df(teams=["A"])
    df = df.sort_values("date", ascending=False)

    splitter = PandasTimeSeriesSplit(
        groupby="team", datetime_col="date", n_splits=3
    )

    with pytest.raises(ValueError):
        for X_idx, y_idx in splitter.split(df):
            pass


def test_different_series_lengths():
    df = pd.concat(
        [
            input_df(teams=["A"], num_days=15),
            input_df(teams=["B"], num_days=30),
        ],
        ignore_index="ignore",
    )

    split_lens = {"A": 3, "B": 6}

    splitter = PandasTimeSeriesSplit(
        groupby="team", datetime_col="date", n_splits=4
    )

    count = 0
    for X_idx, y_idx in splitter.split(df):
        count += 1
        X, y = df.iloc[X_idx], df.iloc[y_idx]

        for team in df.team.unique():
            X_team, y_team = X[X.team == team], y[y.team == team]
            min_train, max_train = X_team["date"].min(), X_team["date"].max()
            min_test, max_test = y_team["date"].min(), y_team["date"].max()

            assert (min_test - max_train).days == 1
            assert (max_train - min_train).days == count * split_lens[team] - 1
            assert (max_test - min_test).days == split_lens[team] - 1

    assert count == 4
