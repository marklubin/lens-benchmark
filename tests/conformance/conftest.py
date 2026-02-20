"""Conformance test fixtures for adapter contract testing."""
from __future__ import annotations

import pytest

from lens.adapters.registry import get_adapter

_ADAPTER_NAMES = ["null", "sqlite", "sqlite-fts", "compaction"]


@pytest.fixture(params=_ADAPTER_NAMES)
def adapter_instance(request):
    """Instantiate the adapter under test."""
    cls = get_adapter(request.param)
    return cls()


@pytest.fixture(params=_ADAPTER_NAMES)
def adapter_name(request):
    """The name of the adapter being tested."""
    return request.param
