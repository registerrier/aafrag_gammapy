from aafrag_gammapy.models import AafragGammaSpectralModel, PowerLawParticleDistribution


def test_parameter_aggregation():
    pl_p = PowerLawParticleDistribution(amplitude="1e40 TeV-1", index=2.0)
    pl_he = PowerLawParticleDistribution(amplitude="1e39 TeV-1", index=2.3)
    pl_he.index.frozen = True

    model = AafragGammaSpectralModel({"p": pl_p, "He": pl_he})

    names = [p.name for p in model.parameters]
    # 3 params each from pl_p and pl_he (amplitude, index, reference), plus n_H
    assert len(model.parameters) == 3 + 3 + 1
    assert names.count("index") == 2  # same name from two submodels, no collision error
    assert "n_H" in names

    # the aggregated Parameters are the *same objects* as the submodels' own -- so
    # frozen/free flags (and any future fit) are shared by reference, not copied.
    assert pl_he.index in model.parameters
    assert pl_p.index in model.parameters
    assert pl_he.index.frozen is True
    assert pl_p.index.frozen is False


def test_parameter_aggregation_single_species():
    pl = PowerLawParticleDistribution(amplitude="1e40 TeV-1", index=2.0)
    model = AafragGammaSpectralModel(pl)

    names = [p.name for p in model.parameters]
    assert len(model.parameters) == 4  # amplitude, index, reference, n_H
    assert names == ["amplitude", "index", "reference", "n_H"]
