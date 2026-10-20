# climada-dsge-cotedivoire

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![CLIMADA 4.0+](https://img.shields.io/badge/CLIMADA-4.0+-green.svg)](https://github.com/CLIMADA-project/climada_python)

**CLIMADA-DSGE-AIN-CGE** : a structural Bayesian DSGE framework linking CLIMADA physical damages to macroeconomic transmission in Cote d'Ivoire.

## What this does

This package extends [CLIMADA](https://github.com/CLIMADA-project/climada_python) beyond its native scope by converting physical climate damages into macroeconomic shocks:
CLIMADA Hazard -> Omega_t (climate productivity) -> DSGE -> macro multiplier -> adaptation NPV

text

It is **complementary** to `climada_python` (Core) and `climada_petals` (sectoral modules): it **consumes** their outputs and **adds** four analytical layers absent from the official ecosystem:

| Module | Function | Status |
|---|---|---|
| `OmegaFactor` | Builds `Omega_t` from SODEXAM + CLIMADA hazards | Wraps `climada.hazard.Hazard` |
| `LinearDSGE` | Dynare-Python bridge via `pydsge` | Estimated (MCMC, 50k draws) |
| `GPDFitter` | POT/GPD extreme value analysis | Estimated (MLE) |
| `AdaptiveINetwork` | Adaptive input-output network | **Calibrated** (INS 2021) |
| `AdaptationNPV` | SLR damage + adaptation NPV | **Conditional simulation** |

## Installation

```bash
pip install git+https://github.com/zagbo-dayory/climada-dsge-cotedivoire.git
Requires climada>=4.0, pydsge>=0.4, numpy, pandas, scipy, Python >=3.10.

Quickstart
bash
# 1) Smoke test (no external data required - generates everything)
python climada_dsge.py --smoke-test

# 2) Full pipeline
python climada_dsge.py --full

# 3) Export Dynare (.mod + JSON posterior)
python climada_dsge.py --export-dynare-json
Or in Python:

python
from climada_dsge import ZagboPipeline

pipeline = ZagboPipeline(data_dir="data", output_dir="outputs", seed=42)
pipeline.run()
pipeline.export_tables()
Documentation
Paper: MPRA 128349

Dynare model: generated at outputs/climada_dsge_civ.mod

Replication notes: see Replication Note M2 in the paper for the Python-vs-Dynare multiplier discrepancy.

Data availability
Institutional data (DPPSE, DGBF, SODEXAM, BNETD) are available on request per data-sharing agreements. Synthetic data calibrated to published posterior moments are generated automatically by generate_synthetic_data() for CI and smoke tests.

Citation
bibtex
@techreport{zagbo2025climada,
  title  = {Climate Risk and Macroeconomic Vulnerability in Cote d'Ivoire},
  author = {Zagbo, D. D. D.},
  year   = {2025},
  number = {128349},
  institution = {MPRA},
  url    = {https://mpra.ub.uni-muenchen.de/128349/}
}
License
Apache 2.0 - see LICENSE.
