"""gammapy-facing model classes wrapping aafragpy hadronic interaction tables.

Owns primary_composition/target_composition/n_H/distance, parameter aggregation, and
the evaluate() double loop over composition species. Only calls into kernel.py -- no
aafragpy import here (ADR-003).
"""

import numpy as np
from astropy import constants as const
from astropy import units as u

from gammapy.modeling import Parameter, Parameters
from gammapy.modeling.models import SPECTRAL_MODEL_REGISTRY, ModelBase, SpectralModel
from gammapy.utils.registry import Registry

from . import kernel

__all__ = [
    "ParticleDistribution",
    "PowerLawParticleDistribution",
    "ExpCutoffPowerLawParticleDistribution",
    "PARTICLE_DISTRIBUTION_REGISTRY",
    "AafragSpectralModelBase",
    "AafragGammaSpectralModel",
    "AafragNuESpectralModel",
    "AafragAntiNuESpectralModel",
    "AafragNuMuSpectralModel",
    "AafragAntiNuMuSpectralModel",
]

#: Number of log-spaced points in the internal primary-energy integration grid
#: (ADR-018). Not user-configurable in v1 -- revisit via 06_performance_benchmark.ipynb
#: (Step 6) if it turns out to need tuning.
N_PRIMARY_GRID_POINTS = 100

#: ADR-015: kernel.py speaks aafragpy's native target tags; only "H" (the element-symbol
#: spelling used by ADR-007's user-facing target_composition convention) needs
#: translating to aafragpy's "p" (a proton/hydrogen target).
_TARGET_ALIASES = {"H": "p"}

#: Differential flux unit aafragpy's mb-cross-section output converts into (ADR-018).
_FLUX_UNIT = u.cm**-2 * u.s**-1 * u.GeV**-1

#: 1 millibarn in cm^2, used to convert aafragpy's mb cross-sections (ADR-018).
_MB = u.mbarn


def _translate_target_species(species):
    """Map a user-facing target_composition key to aafragpy's native tag (ADR-015)."""
    return _TARGET_ALIASES.get(species, species)


class ParticleDistribution(ModelBase):
    """Base class for cosmic-ray particle distributions dN/dE [energy^-1] (ADR-019).

    Represents the total, already volume-integrated particle spectrum of a
    cosmic-ray population in the emitting region -- the same convention `naima`'s
    radiative models use for `particle_distribution`. Deliberately *not* a
    `~gammapy.modeling.models.SpectralModel`: every gammapy `SpectralModel`
    amplitude-like `Parameter` is hardcoded to a flux dimension
    (`cm^-2 s^-1 TeV^-1`), which cannot represent a bare `energy^-1` particle count
    (ADR-019 supersedes ADR-005/ADR-006, which had assumed `SpectralModel` reuse for
    this role). Still reuses gammapy's `Parameter`/`Parameters`/`ModelBase`
    machinery directly, so instances remain fittable, support `frozen`, etc.
    """

    _type = "particle_distribution"

    def __call__(self, energy):
        kwargs = {par.name: par.quantity for par in self.parameters}
        return self.evaluate(energy, **kwargs)


class PowerLawParticleDistribution(ParticleDistribution):
    """Power-law cosmic-ray particle distribution dN/dE = amplitude * (E/reference)^-index."""

    tag = ["PowerLawParticleDistribution", "pl-particle"]

    amplitude = Parameter("amplitude", "1e40 TeV-1")
    index = Parameter("index", 2.0)
    reference = Parameter("reference", "1 TeV", frozen=True)

    @staticmethod
    def evaluate(energy, amplitude, index, reference):
        return amplitude * (energy / reference) ** (-index)


class ExpCutoffPowerLawParticleDistribution(ParticleDistribution):
    """Power law with exponential cutoff: dN/dE = PowerLaw(E) * exp(-E * lambda_)."""

    tag = ["ExpCutoffPowerLawParticleDistribution", "ecpl-particle"]

    amplitude = Parameter("amplitude", "1e40 TeV-1")
    index = Parameter("index", 2.0)
    reference = Parameter("reference", "1 TeV", frozen=True)
    lambda_ = Parameter("lambda_", "0 TeV-1")

    @staticmethod
    def evaluate(energy, amplitude, index, reference, lambda_):
        pwl = amplitude * (energy / reference) ** (-index)
        cutoff = np.exp(-(energy * lambda_).to_value(u.dimensionless_unscaled))
        return pwl * cutoff


#: Package-local tag->class lookup for `ParticleDistribution` subclasses (ADR-020), used by
#: `AafragSpectralModelBase.from_dict` to rebuild `primary_composition` submodels. Distinct
#: from gammapy's `SPECTRAL_MODEL_REGISTRY`: `ParticleDistribution` isn't a `SpectralModel`
#: (ADR-019), so gammapy's own YAML system has no reason to know about it.
PARTICLE_DISTRIBUTION_REGISTRY = Registry(
    [PowerLawParticleDistribution, ExpCutoffPowerLawParticleDistribution]
)


class AafragSpectralModelBase(SpectralModel):
    """Shared base class for aafragpy-wrapped hadronic-interaction spectral models.

    Computes the secondary (gamma-ray or neutrino) differential flux produced by an
    arbitrary-composition cosmic-ray population interacting with an arbitrary-
    composition target medium, via aafragpy's AAfrag cross-section tables
    (DECISIONS.md ADR-003/007/018/019).

    Parameters
    ----------
    primary_composition : `ParticleDistribution` or dict[str, `ParticleDistribution`]
        Cosmic-ray primary species -> particle distribution dN/dE (ADR-019). A bare
        `ParticleDistribution` (not a dict) is shorthand for
        ``{"p": primary_composition}``.
    target_composition : dict[str, float], optional
        Target species -> abundance relative to `n_H` (ADR-007). Keys are element
        symbols (``"H"``, ``"He"``, ``"C"``); ``"H"`` is translated to aafragpy's
        native ``"p"`` tag internally (ADR-015). Defaults to ``{"H": 1.0}``.
    n_H : `~astropy.units.Quantity`, optional
        Target medium reference number density. The sole fitted `Parameter` besides
        whatever `primary_composition` submodels contribute (ADR-008). Default is
        ``1 * u.cm**-3``.
    distance : `~astropy.units.Quantity`, optional
        Distance to the source, used to convert the emitted differential luminosity
        into a flux at Earth (ADR-018). Not a fitted `Parameter`, matching
        `~gammapy.modeling.models.NaimaSpectralModel`'s convention. Default is
        ``1 * u.kpc``.
    """

    #: aafragpy secondary-channel tag (one of kernel.CHANNELS); set by the
    #: generated subclasses below (ADR-004).
    channel = None

    def __init__(self, primary_composition, target_composition=None, n_H=1 * u.cm**-3,
                 distance=1 * u.kpc):
        if isinstance(primary_composition, ParticleDistribution):
            primary_composition = {"p": primary_composition}
        if not primary_composition:
            raise ValueError("primary_composition must be non-empty")
        self.primary_composition = dict(primary_composition)

        if target_composition is None:
            target_composition = {"H": 1.0}
        if not target_composition:
            raise ValueError("target_composition must be non-empty")
        self.target_composition = dict(target_composition)
        self._target_tags = {
            species: _translate_target_species(species)
            for species in self.target_composition
        }

        # ADR-009: validate every (primary, target) pair at construction time, before
        # any Dataset/Fit object is touched.
        pairs = [
            (primary_species, target_tag)
            for primary_species in self.primary_composition
            for target_tag in self._target_tags.values()
        ]
        for primary_species, target_tag in pairs:
            kernel.validate_species(primary_species, target_tag)

        # ADR-018: shared primary-energy integration grid = intersection of every
        # (primary, target) pair's valid threshold range.
        ranges = [kernel.energy_range(p, t, self.channel) for p, t in pairs]
        e_min = max(lo for lo, _ in ranges)
        e_max = min(hi for _, hi in ranges)
        if not e_min < e_max:
            raise ValueError(
                "No overlapping valid primary-energy range for this composition: "
                f"intersection is [{e_min}, {e_max}] GeV"
            )
        self._energy_primary_gev = np.geomspace(e_min, e_max, N_PRIMARY_GRID_POINTS)

        self.distance = u.Quantity(distance)

        n_H_parameter = n_H if isinstance(n_H, Parameter) else Parameter("n_H", u.Quantity(n_H))
        self.default_parameters = Parameters([n_H_parameter])

        super().__init__()

    @property
    def parameters(self):
        params = Parameters([])
        for submodel in self.primary_composition.values():
            params = params + submodel.parameters
        return params + Parameters([self.n_H])

    def _secondary_energy_gev(self):
        # ADR-014: the secondary-energy grid is identical across every (primary,
        # target) pair for a fixed channel, so any one pair's cross_section() call
        # gives the grid shared by the whole composition.
        primary_species = next(iter(self.primary_composition))
        target_tag = next(iter(self._target_tags.values()))
        cs = kernel.cross_section(
            primary_species, target_tag, self.channel, self._energy_primary_gev
        )
        return cs.energy_secondary

    def _bare_primary_flux(self, dnde_quantity):
        # dnde_quantity is a genuine dN/dE Quantity from a ParticleDistribution
        # (energy^-1, e.g. TeV^-1) -- a real unit conversion, not a reinterpretation.
        return dnde_quantity.to_value(u.GeV**-1)

    def _secondary_flux(self, energy, primary_fluxes_bare, n_H_quantity):
        target_composition = {
            self._target_tags[species]: abundance
            for species, abundance in self.target_composition.items()
        }
        raw = kernel.combine_species(
            primary_fluxes_bare,
            target_composition,
            n_H_quantity.to_value(u.cm**-3),
            self.channel,
            self._energy_primary_gev,
        )
        # ADR-018 dimensional derivation: raw carries implicit units
        # [mbarn * cm^-3 / GeV] (n_H times a cross-section/GeV-primary-integrated
        # sum); multiplying by c and dividing by 4 pi distance^2 gives a proper
        # differential flux -- .to() below both performs and verifies that.
        raw_quantity = raw * (_MB * u.cm**-3 / u.GeV)
        secondary_flux = (const.c * raw_quantity / (4 * np.pi * self.distance**2)).to(
            _FLUX_UNIT
        )

        energy_secondary_gev = self._secondary_energy_gev()
        energy_gev = kernel.energy_to_gev(energy)
        interpolated = _interp_log_log(
            energy_secondary_gev, secondary_flux.to_value(_FLUX_UNIT), energy_gev
        )
        unit = 1 / (energy.unit * u.cm**2 * u.s)
        result = (interpolated * _FLUX_UNIT).to(unit)
        return result.reshape(energy.shape)

    def __call__(self, energy):
        primary_energy = kernel.gev_to_energy(self._energy_primary_gev)
        primary_fluxes_bare = {
            species: self._bare_primary_flux(submodel(primary_energy))
            for species, submodel in self.primary_composition.items()
        }
        return self._secondary_flux(energy, primary_fluxes_bare, self.n_H.quantity)

    def evaluate(self, energy, *args):
        primary_energy = kernel.gev_to_energy(self._energy_primary_gev)
        primary_fluxes_bare = {}
        offset = 0
        for species, submodel in self.primary_composition.items():
            n_params = len(submodel.parameters)
            sub_args = args[offset : offset + n_params]
            offset += n_params
            flux = submodel.evaluate(primary_energy, *sub_args)
            primary_fluxes_bare[species] = self._bare_primary_flux(flux)
        n_H_quantity = u.Quantity(args[offset])
        return self._secondary_flux(energy, primary_fluxes_bare, n_H_quantity)

    def to_dict(self, full_output=False):
        """Create dictionary for YAML serialisation (ADR-020)."""
        tag = self.tag[0] if isinstance(self.tag, list) else self.tag
        primary_composition = {
            species: submodel.to_dict(full_output)["particle_distribution"]
            for species, submodel in self.primary_composition.items()
        }
        data = {
            "type": tag,
            "primary_composition": primary_composition,
            "target_composition": dict(self.target_composition),
            "n_H": Parameters([self.n_H]).to_dict()[0],
            "distance": {
                "value": float(self.distance.value),
                "unit": self.distance.unit.to_string("fits"),
            },
        }
        return {"spectral": data}

    @classmethod
    def from_dict(cls, data, **kwargs):
        data = data["spectral"]
        if data["type"] not in cls.tag:
            raise ValueError(
                f"Invalid model type {data['type']} for class {cls.__name__}"
            )

        primary_composition = {}
        for species, submodel_data in data["primary_composition"].items():
            submodel_cls = PARTICLE_DISTRIBUTION_REGISTRY.get_cls(submodel_data["type"])
            primary_composition[species] = submodel_cls.from_dict(submodel_data)

        n_H = Parameters.from_dict([data["n_H"]])[0]
        distance = u.Quantity(data["distance"]["value"], data["distance"]["unit"])

        return cls(
            primary_composition,
            target_composition=dict(data["target_composition"]),
            n_H=n_H,
            distance=distance,
        )


def _interp_log_log(x_ref, y_ref, x_query):
    """Log-log interpolate y_ref(x_ref) onto x_query (ADR-018).

    Flat extrapolation outside [x_ref.min(), x_ref.max()] (numpy.interp's default),
    not physically exact but a safe default for v1. y_ref values <= 0 are floored
    before taking the log to avoid -inf.
    """
    y_ref_safe = np.clip(y_ref, a_min=1e-300, a_max=None)
    log_y = np.interp(np.log(x_query), np.log(x_ref), np.log(y_ref_safe))
    return np.exp(log_y)


#: (class_name, aafragpy channel tag, human-readable label) -- ADR-004: generates one
#: real, module-level, individually-importable class per channel at import time.
_CHANNEL_SPECS = [
    ("AafragGammaSpectralModel", "gam", "Gamma-ray"),
    ("AafragNuESpectralModel", "nu_e", "Electron-neutrino"),
    ("AafragAntiNuESpectralModel", "anu_e", "Electron-antineutrino"),
    ("AafragNuMuSpectralModel", "nu_mu", "Muon-neutrino"),
    ("AafragAntiNuMuSpectralModel", "anu_mu", "Muon-antineutrino"),
]

for _class_name, _channel_tag, _label in _CHANNEL_SPECS:
    _cls = type(
        _class_name,
        (AafragSpectralModelBase,),
        {
            "channel": _channel_tag,
            "tag": [_class_name, f"aafrag-{_channel_tag}"],
            "__doc__": (
                f"{_label} spectral model from hadronic pion/kaon decay, via "
                "aafragpy/AAfrag. See `AafragSpectralModelBase` for parameters."
            ),
        },
    )
    globals()[_class_name] = _cls
    # ADR-020: makes this class reachable from `Models.from_yaml`/`SkyModel.from_dict` via
    # SPECTRAL_MODEL_REGISTRY.get_cls(tag) -- without this, from_dict on the class itself
    # works, but a full Models YAML round-trip cannot find the class at all.
    SPECTRAL_MODEL_REGISTRY.append(_cls)

del _class_name, _channel_tag, _label, _cls
