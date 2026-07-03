"""Gammapy-agnostic wrapper around aafragpy.

Owns GeV <-> astropy.units.Quantity conversion, species validation, caching of
aafragpy.get_cross_section matrices, and the get_spectrum-based convolution /
multi-species combination. This is the only module that imports aafragpy or
depends on any of its return-value quirks (see DECISIONS.md ADR-011/012/014).
"""

from typing import NamedTuple

import numpy as np
from astropy import units as u

import aafragpy

__all__ = [
    "UnsupportedSpeciesError",
    "SUPPORTED_PAIRS",
    "CHANNELS",
    "validate_species",
    "energy_to_gev",
    "gev_to_energy",
    "CrossSection",
    "cross_section",
    "convolve_spectrum",
    "combine_species",
]


class UnsupportedSpeciesError(ValueError):
    """Raised when a (primary, target) species pair isn't tabulated by aafragpy."""


# ADR-012: AAfrag tabulates exactly these 8 (primary, target) pairs -- not a free
# cross product of independently-supported primary/target species lists.
SUPPORTED_PAIRS = frozenset(
    {
        ("p", "p"),
        ("p", "He"),
        ("p", "C"),
        ("He", "p"),
        ("He", "He"),
        ("Al", "p"),
        ("C", "p"),
        ("Fe", "p"),
    }
)

# ADR-011: real aafragpy secondary-channel tag strings for the v1 in-scope channels
# (gamma-ray + all four neutrino flavors). The get_cross_section docstring's channel
# list is stale -- these were confirmed against aafragpy.open_data_files source.
CHANNELS = ("gam", "nu_e", "anu_e", "nu_mu", "anu_mu")


def validate_species(primary, target):
    """Raise UnsupportedSpeciesError unless (primary, target) is tabulated (ADR-009/012)."""
    if (primary, target) not in SUPPORTED_PAIRS:
        raise UnsupportedSpeciesError(
            f"aafragpy has no tabulated data for primary={primary!r}, "
            f"target={target!r}. Supported (primary, target) pairs: "
            f"{sorted(SUPPORTED_PAIRS)}"
        )


def energy_to_gev(energy):
    """Convert a gammapy/astropy energy Quantity to the bare GeV floats aafragpy expects."""
    return np.atleast_1d(energy.to_value(u.GeV)).astype(float)


def gev_to_energy(energy_gev):
    """Tag a bare-GeV array returned by aafragpy as an astropy Quantity."""
    return np.asarray(energy_gev, dtype=float) * u.GeV


class CrossSection(NamedTuple):
    """A differential cross-section matrix and the energy grids it's defined on."""

    matrix: np.ndarray  # mb/GeV, shape (n_primary, n_secondary)
    energy_primary: np.ndarray  # GeV
    energy_secondary: np.ndarray  # GeV


_cross_section_cache = {}


def cross_section(primary, target, channel, energy_primary_gev):
    """Cross-section matrix for (primary, target, channel) on a primary energy grid.

    Computed once per (primary, target, channel, energy_primary_gev) key and memoized
    for the process lifetime (ADR-003/ADR-013: this is the ~500x more expensive call,
    aafragpy itself does not cache it). aafragpy's default secondary-energy binning is
    always used (E_secondaries left unset) -- ADR-014 confirms that default grid is
    identical across every supported (primary, target) pair for a given channel, which
    is what lets combine_species sum contributions without interpolation.
    """
    validate_species(primary, target)
    energy_primary_gev = np.asarray(energy_primary_gev, dtype=float)
    key = (primary, target, channel, tuple(energy_primary_gev))
    cached = _cross_section_cache.get(key)
    if cached is None:
        matrix, e_primary, e_secondary = aafragpy.get_cross_section(
            channel, f"{primary}-{target}", E_primaries=energy_primary_gev
        )
        cached = CrossSection(matrix, np.atleast_1d(e_primary), np.asarray(e_secondary))
        _cross_section_cache[key] = cached
    return cached


def convolve_spectrum(cs, primary_flux):
    """Convolve a primary flux array (on cs.energy_primary) against cs.matrix.

    Thin wrapper around aafragpy.get_spectrum. Returns the secondary differential
    spectrum on cs.energy_secondary.
    """
    return aafragpy.get_spectrum(
        cs.energy_primary, cs.energy_secondary, cs.matrix, np.asarray(primary_flux)
    )


def combine_species(primary_fluxes, target_composition, n_H, channel, energy_primary_gev):
    """Weighted double loop over primary x target species (ADR-006/ADR-007).

    Parameters
    ----------
    primary_fluxes : dict[str, numpy.ndarray]
        Primary species -> flux array already evaluated on energy_primary_gev.
    target_composition : dict[str, float]
        Target species -> abundance relative to n_H.
    n_H : float
        Reference target number density.
    channel : str
        A single aafragpy secondary-channel tag (see CHANNELS) applied to every
        species pair in this call -- never mix channels in one combine_species call
        (ADR-014).
    energy_primary_gev : numpy.ndarray
        Primary energy grid (GeV) that every array in primary_fluxes is evaluated on.

    Returns
    -------
    numpy.ndarray
        Secondary differential spectrum, on the shared secondary-energy grid.
    """
    if not primary_fluxes or not target_composition:
        raise ValueError("primary_fluxes and target_composition must both be non-empty")

    total = None
    for primary_species, primary_flux in primary_fluxes.items():
        for target_species, abundance in target_composition.items():
            cs = cross_section(primary_species, target_species, channel, energy_primary_gev)
            contribution = n_H * abundance * convolve_spectrum(cs, primary_flux)
            total = contribution if total is None else total + contribution
    return total
