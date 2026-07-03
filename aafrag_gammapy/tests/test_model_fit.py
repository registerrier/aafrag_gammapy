from astropy import units as u

from gammapy.datasets import SpectrumDataset
from gammapy.maps import MapAxis, RegionGeom
from gammapy.modeling import Fit
from gammapy.modeling.models import SkyModel

from aafrag_gammapy.models import AafragGammaSpectralModel, PowerLawParticleDistribution


def test_fit_smoke():
    # Smoke test only (per CLAUDE.md Step 5): confirms a minimal gammapy Fit runs to
    # completion without exception -- not a numerical-accuracy check (that belongs in
    # the Step 6 validation notebooks). Parameter values below are chosen purely to
    # produce a non-degenerate number of counts, not to be physically realistic.
    energy_axis = MapAxis.from_energy_bounds("1 TeV", "100 TeV", nbin=5, name="energy")
    energy_axis_true = MapAxis.from_energy_bounds(
        "1 TeV", "100 TeV", nbin=5, name="energy_true"
    )
    geom = RegionGeom.create("icrs;circle(0,0,0.1)", axes=[energy_axis])

    dataset = SpectrumDataset.create(geom=geom, energy_axis_true=energy_axis_true)
    dataset.exposure.data += 1e8
    dataset.mask_safe.data[...] = True

    primary = PowerLawParticleDistribution(amplitude="1e46 TeV-1", index=2.2)
    spectral_model = AafragGammaSpectralModel(
        primary, n_H=1e3 * u.cm**-3, distance=0.1 * u.kpc
    )
    sky_model = SkyModel(spectral_model=spectral_model, name="test-source")
    dataset.models = [sky_model]
    dataset.fake(random_state=0)

    assert dataset.counts.data.sum() > 0

    fit = Fit()
    result = fit.run([dataset])  # must not raise

    assert result is not None
