import itertools

import pytest

from aafrag_gammapy.kernel import (
    SUPPORTED_PAIRS,
    UnsupportedSpeciesError,
    energy_range,
    validate_species,
)

PRIMARIES = ["p", "He", "Al", "C", "Fe"]
TARGETS = ["p", "He", "C"]

INVALID_PAIRS = [
    (primary, target)
    for primary, target in itertools.product(PRIMARIES, TARGETS)
    if (primary, target) not in SUPPORTED_PAIRS
]


@pytest.mark.parametrize("primary,target", sorted(SUPPORTED_PAIRS))
def test_species_validation_accepts_supported_pairs(primary, target):
    validate_species(primary, target)  # must not raise


@pytest.mark.parametrize(
    "primary,target", INVALID_PAIRS + [("Xx", "p"), ("p", "Xx"), ("H", "p")]
)
def test_species_validation_raises(primary, target):
    with pytest.raises(UnsupportedSpeciesError):
        validate_species(primary, target)


@pytest.mark.parametrize("primary,target", sorted(SUPPORTED_PAIRS))
def test_energy_range_is_positive_and_ordered(primary, target):
    lo, hi = energy_range(primary, target, "gam")
    assert 0 < lo < hi


def test_energy_range_matches_known_thresholds():
    # Regression pin against the confirmed values from ADR-012 (in GeV).
    assert energy_range("p", "p", "gam")[0] == pytest.approx(4.0)
    assert energy_range("p", "C", "gam")[0] == pytest.approx(999.4, rel=1e-3)
    assert energy_range("He", "p", "gam")[0] == pytest.approx(5.0)
