# Data directory

This directory holds the input data used by `climada_dsge.py`.

## Synthetic data (auto-generated)

Running `python climada_dsge.py --smoke-test` generates:

- `dppse_synthetic.csv` - 92 quarters of macro observables
- `SAM_synthetic_12sectors.csv` - 12x12 social accounting matrix
- `climada_damages.csv` - 2000 fat-tailed damage draws (xi ~ 0.278)

These files are calibrated to the posterior moments in Table 2 of
Zagbo (2025, MPRA 128349). They reproduce the sign and mechanism of the
climate multiplier but not its exact numerical value (see Replication
Note M2 in the paper).

## Institutional data (on request)

The following series are not included in this repository due to
data-sharing agreements. They are available on request:

- DPPSE/DGBF: quarterly macro-fiscal series, 2000Q1-2022Q4
- BCEAO: policy rate, monetary aggregates, 2000Q1-2022Q4
- SODEXAM: daily precipitation and temperature (5 stations), 2000-2022
- BNETD: LIDAR coastal surveys (50 cm resolution, 400 km), 2019-2022
- DPPSE 2022 SAM: 12-sector social accounting matrix

Contact: dayord8@gmail.com
