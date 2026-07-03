import pytest

from aafrag_gammapy import kernel


@pytest.fixture(autouse=True)
def clear_cross_section_cache():
    kernel._cross_section_cache.clear()
    yield
    kernel._cross_section_cache.clear()
