import pytest
from pydantic import ValidationError

from app.core.config import Settings


@pytest.mark.parametrize("value", [0, 1, 4, -10])
def test_index_refresh_interval_rejects_too_small(value):
    with pytest.raises(ValidationError):
        Settings(index_refresh_interval_sec=value)


@pytest.mark.parametrize("value", [5, 60])
def test_index_refresh_interval_accepts_reasonable(value):
    assert Settings(index_refresh_interval_sec=value).index_refresh_interval_sec == value
