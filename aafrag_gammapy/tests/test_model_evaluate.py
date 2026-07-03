import numpy as np
import pytest
from astropy import units as u

from aafrag_gammapy import models

CHANNEL_CLASSES = [
    models.AafragGammaSpectralModel,
    models.AafragNuESpectralModel,
    models.AafragAntiNuESpectralModel,
    models.AafragNuMuSpectralModel,
    models.AafragAntiNuMuSpectralModel,
]


def _pl():
    return models.PowerLawParticleDistribution(amplitude="1e40 TeV-1", index=2.2)


def test_evaluate_shape_and_units_array():
    model = models.AafragGammaSpectralModel(_pl())
    energy = np.geomspace(0.1, 100, 7) * u.TeV

    flux = model(energy)

    assert flux.shape == energy.shape
    assert flux.unit.is_equivalent(u.Unit("1 / (cm2 s TeV)"))
    assert np.all(flux.value >= 0)


def test_evaluate_shape_and_units_scalar():
    model = models.AafragGammaSpectralModel(_pl())

    flux = model(1 * u.TeV)

    assert flux.isscalar
    assert flux.unit.is_equivalent(u.Unit("1 / (cm2 s TeV)"))
    assert flux.value >= 0


def test_evaluate_matches_positional_evaluate():
    model = models.AafragGammaSpectralModel(_pl())
    energy = np.geomspace(0.1, 100, 5) * u.TeV

    called = model(energy)
    args = [p.quantity for p in model.parameters]
    evaluated = model.evaluate(energy, *args)

    np.testing.assert_allclose(called.value, evaluated.value)


@pytest.mark.parametrize("model_cls", CHANNEL_CLASSES)
def test_channel_variants_tag_and_nonnegative(model_cls):
    model = model_cls(_pl())
    energy = np.geomspace(0.1, 100, 5) * u.TeV

    assert model.tag[0] == model_cls.__name__

    flux = model(energy)
    assert np.all(flux.value >= 0)


def test_channel_variants_numerically_distinguishable():
    pl = _pl()
    energy = np.geomspace(1, 10, 5) * u.TeV

    fluxes = [
        model_cls(pl)(energy).to_value(u.Unit("1/(cm2 s TeV)")) for model_cls in CHANNEL_CLASSES
    ]

    for i in range(len(fluxes)):
        for j in range(i + 1, len(fluxes)):
            # atol=0: these fluxes are ~1e-21, so the default atol=1e-8 would make
            # np.allclose call any two tiny-but-very-different arrays "close".
            assert not np.allclose(fluxes[i], fluxes[j], atol=0), (
                f"{CHANNEL_CLASSES[i].__name__} and {CHANNEL_CLASSES[j].__name__} "
                "produced identical flux -- possible wrong-channel-string bug"
            )


def test_default_single_species_convenience():
    pl = _pl()
    energy = np.geomspace(0.1, 100, 5) * u.TeV

    model_bare = models.AafragGammaSpectralModel(pl)
    model_dict = models.AafragGammaSpectralModel({"p": pl})

    np.testing.assert_allclose(
        model_bare(energy).to_value(u.Unit("1/(cm2 s TeV)")),
        model_dict(energy).to_value(u.Unit("1/(cm2 s TeV)")),
    )
