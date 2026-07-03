# aafrag_gammapy

`gammapy.modeling.models.SpectralModel` wrappers around
[`aafragpy`](https://github.com/aafragpy/aafragpy) (a Python implementation of the AAFrag
hadronic interaction tables), for pion/kaon-decay gamma-ray and neutrino production from
cosmic-ray-nucleus interactions with an arbitrary-composition target medium.

This package is under active design/implementation. See [`CLAUDE.md`](CLAUDE.md) for the
full architecture, scope, and implementation plan, and [`DECISIONS.md`](DECISIONS.md) for
the rationale behind every design choice.

## Install (development)

```console
pip install -e ".[test]"
```

## Test

```console
pytest
# or, for a reproducible isolated-environment run:
tox
```

## Citing

This package is a thin wrapper — the physics comes entirely from AAfrag/`aafragpy`. If you
use it, please cite the underlying AAfrag work:

> S. Koldobskiy, M. Kachelrieß, A. Lskavyan, A. Neronov, S. Ostapchenko, and D. V. Semikoz,
> "Energy spectra of secondaries in proton-proton interactions," Phys. Rev. D, vol. 104,
> no. 12, p. 123027, 2021.
> [DOI: 10.1103/PhysRevD.104.123027](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.104.123027),
> [arXiv:2110.00496](https://arxiv.org/abs/2110.00496).

See the [`aafragpy` repository](https://github.com/aafragpy/aafragpy) for the full
reference list, including the original Fortran AAfrag code and the low-energy
parameterizations (Kamae et al. 2006, Kafexhiu et al. 2014).
