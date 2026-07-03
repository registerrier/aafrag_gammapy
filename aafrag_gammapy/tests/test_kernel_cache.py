import numpy as np

import aafragpy
from aafrag_gammapy import kernel


def _counting_get_cross_section(monkeypatch):
    calls = {"n": 0}
    real_get_cross_section = aafragpy.get_cross_section

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real_get_cross_section(*args, **kwargs)

    monkeypatch.setattr(aafragpy, "get_cross_section", counting)
    return calls


def test_cache_reuses_matrix(monkeypatch):
    calls = _counting_get_cross_section(monkeypatch)
    energy_grid = np.geomspace(1e3, 1e5, 5)

    cs1 = kernel.cross_section("p", "p", "gam", energy_grid)
    cs2 = kernel.cross_section("p", "p", "gam", energy_grid)

    assert calls["n"] == 1
    assert cs1 is cs2


def test_cache_invalidates_on_grid_change(monkeypatch):
    calls = _counting_get_cross_section(monkeypatch)
    grid_a = np.geomspace(1e3, 1e5, 5)
    grid_b = np.geomspace(2e3, 2e5, 5)

    kernel.cross_section("p", "p", "gam", grid_a)
    kernel.cross_section("p", "p", "gam", grid_b)

    assert calls["n"] == 2


def test_cache_keys_on_channel_too(monkeypatch):
    calls = _counting_get_cross_section(monkeypatch)
    energy_grid = np.geomspace(1e3, 1e5, 5)

    kernel.cross_section("p", "p", "gam", energy_grid)
    kernel.cross_section("p", "p", "nu_e", energy_grid)

    assert calls["n"] == 2
