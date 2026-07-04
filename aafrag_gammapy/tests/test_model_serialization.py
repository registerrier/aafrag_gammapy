import numpy as np
from astropy import units as u
from gammapy.modeling.models import Models, SkyModel

from aafrag_gammapy import models


def test_serialization_roundtrip():
    # ADR-012: only 8 (primary, target) pairs are tabulated at all, so a 2-species
    # primary_composition can only share a target_composition drawn from targets valid
    # for *both* primaries -- {H, He} is the maximal such set (p-p/p-He/He-p/He-He all
    # tabulated; He-C is not). See test_serialization_roundtrip_three_species_target
    # below for the 3-species target_composition case, with a single primary.
    pl_p = models.PowerLawParticleDistribution(amplitude="1e40 TeV-1", index=2.0)
    ecpl_he = models.ExpCutoffPowerLawParticleDistribution(
        amplitude="1e39 TeV-1", index=2.3, lambda_="0.1 TeV-1"
    )
    ecpl_he.index.frozen = True

    model = models.AafragGammaSpectralModel(
        {"p": pl_p, "He": ecpl_he},
        target_composition={"H": 1.0, "He": 0.1},
        n_H="2 cm-3",
        distance="3.5 kpc",
    )
    model.n_H.frozen = True

    sky_model = SkyModel(spectral_model=model, name="test-source")
    original = Models([sky_model])

    # round-trip through a full Models YAML file, not just the model's own to_dict/
    # from_dict -- this is what actually exercises the SPECTRAL_MODEL_REGISTRY/
    # PARTICLE_DISTRIBUTION_REGISTRY lookups (ADR-020).
    restored = Models.from_yaml(original.to_yaml())
    restored_model = restored[0].spectral_model

    assert type(restored_model) is type(model)
    assert restored_model.target_composition == model.target_composition
    assert set(restored_model.primary_composition) == set(model.primary_composition)
    for species, submodel in model.primary_composition.items():
        restored_submodel = restored_model.primary_composition[species]
        assert type(restored_submodel) is type(submodel)
        for par, restored_par in zip(submodel.parameters, restored_submodel.parameters):
            assert restored_par.name == par.name
            assert restored_par.quantity == par.quantity
            assert restored_par.frozen == par.frozen

    assert restored_model.n_H.quantity == model.n_H.quantity
    assert restored_model.n_H.frozen == model.n_H.frozen
    assert restored_model.distance == model.distance

    energy = np.geomspace(0.1, 100, 5) * u.TeV
    np.testing.assert_allclose(
        restored_model(energy).to_value(u.Unit("1/(cm2 s TeV)")),
        model(energy).to_value(u.Unit("1/(cm2 s TeV)")),
    )


def test_serialization_roundtrip_three_species_target():
    pl_p = models.PowerLawParticleDistribution(amplitude="1e40 TeV-1", index=2.1)

    model = models.AafragGammaSpectralModel(
        pl_p, target_composition={"H": 1.0, "He": 0.1, "C": 3e-4}
    )

    sky_model = SkyModel(spectral_model=model, name="test-source")
    original = Models([sky_model])

    restored = Models.from_yaml(original.to_yaml())
    restored_model = restored[0].spectral_model

    assert restored_model.target_composition == model.target_composition

    energy = np.geomspace(0.1, 100, 5) * u.TeV
    np.testing.assert_allclose(
        restored_model(energy).to_value(u.Unit("1/(cm2 s TeV)")),
        model(energy).to_value(u.Unit("1/(cm2 s TeV)")),
    )
