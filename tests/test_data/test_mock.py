import pytest
from timeserio.data import mock


@pytest.mark.parametrize(
    'embed_dim, seq_len', [
        (1, 1),
        (2, 2),
        (0, 1),
        (1, 0),
        (0, 0),
    ]
)
def test_single_user_fit_df_sets_id(embed_dim, seq_len):
    out = mock._single_user_fit_df(embedding_dim=embed_dim, seq_length=seq_len,
                                   id=1)
    assert out.id.nunique() == 1
    assert out.id.unique()[0] == 1
