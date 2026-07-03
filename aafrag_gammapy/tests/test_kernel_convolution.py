import numpy as np

import aafragpy
from aafrag_gammapy import kernel


def test_convolution_matches_aafragpy_tutorial():
    # Regression pin: kernel.convolve_spectrum must reproduce aafragpy.get_spectrum's
    # raw output exactly for identical inputs. (No network access in this environment
    # to fetch aafragpy's published tutorial notebook verbatim; pinned instead against
    # a concrete proton power-law spectrum run directly through aafragpy itself.)
    energy_grid = np.geomspace(1e3, 1e6, 30)
    cs = kernel.cross_section("p", "p", "gam", energy_grid)
    primary_flux = (cs.energy_primary / 1e3) ** -2.7

    expected = aafragpy.get_spectrum(
        cs.energy_primary, cs.energy_secondary, cs.matrix, primary_flux
    )
    actual = kernel.convolve_spectrum(cs, primary_flux)

    np.testing.assert_array_equal(actual, expected)


def test_multi_species_combination():
    # Energy grid within the valid range for every supported combo used below,
    # including the p-C threshold (~999 GeV, ADR-012).
    energy_grid = np.geomspace(2e3, 1e6, 20)
    primary_fluxes = {
        "p": (energy_grid / 1e3) ** -2.0,
        "He": 0.1 * (energy_grid / 1e3) ** -2.3,
    }
    target_composition = {"p": 1.0, "He": 0.1}
    n_H = 3.0

    combined = kernel.combine_species(
        primary_fluxes, target_composition, n_H, "gam", energy_grid
    )

    manual = None
    for primary_species, flux in primary_fluxes.items():
        for target_species, abundance in target_composition.items():
            cs = kernel.cross_section(primary_species, target_species, "gam", energy_grid)
            contribution = n_H * abundance * kernel.convolve_spectrum(cs, flux)
            manual = contribution if manual is None else manual + contribution

    np.testing.assert_allclose(combined, manual)


def test_multi_species_combination_not_double_counted():
    # A single-species, single-target combination must equal one direct convolve call
    # -- catches accidental double-counting in the double loop.
    energy_grid = np.geomspace(2e3, 1e6, 20)
    flux = (energy_grid / 1e3) ** -2.0

    combined = kernel.combine_species({"p": flux}, {"p": 1.0}, 1.0, "gam", energy_grid)

    cs = kernel.cross_section("p", "p", "gam", energy_grid)
    direct = kernel.convolve_spectrum(cs, flux)

    np.testing.assert_allclose(combined, direct)
