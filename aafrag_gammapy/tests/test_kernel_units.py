import numpy as np
import pytest
from astropy import units as u

from aafrag_gammapy import kernel


@pytest.mark.parametrize("unit", [u.TeV, u.GeV, u.MeV])
def test_units_roundtrip(unit):
    energy = np.array([1.0, 10.0, 100.0]) * unit

    gev = kernel.energy_to_gev(energy)
    roundtrip = kernel.gev_to_energy(gev).to(unit)

    assert np.allclose(roundtrip.value, energy.value)


def test_energy_to_gev_returns_array_for_scalar():
    gev = kernel.energy_to_gev(1 * u.TeV)
    assert isinstance(gev, np.ndarray)
    assert gev.shape == (1,)
    assert np.isclose(gev[0], 1000.0)
