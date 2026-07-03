import itertools

import pytest

from aafrag_gammapy.kernel import SUPPORTED_PAIRS, UnsupportedSpeciesError, validate_species

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
