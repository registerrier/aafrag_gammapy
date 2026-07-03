import pytest

from aafrag_gammapy import kernel, models


def _pl():
    return models.PowerLawParticleDistribution(amplitude="1e40 TeV-1", index=2.2)


def test_unsupported_primary_species_fails_before_fit():
    with pytest.raises(kernel.UnsupportedSpeciesError):
        models.AafragGammaSpectralModel({"Xx": _pl()})


def test_unsupported_target_species_fails_before_fit():
    with pytest.raises(kernel.UnsupportedSpeciesError):
        models.AafragGammaSpectralModel(_pl(), target_composition={"Xx": 1.0})


def test_empty_primary_composition_raises():
    with pytest.raises(ValueError):
        models.AafragGammaSpectralModel({})


def test_empty_target_composition_raises():
    with pytest.raises(ValueError):
        models.AafragGammaSpectralModel(_pl(), target_composition={})
