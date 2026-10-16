# Paper 2027 — Reproducible bike-sharing interval audit

This package accompanies the *Transportation Research Record* manuscript **When Aggregate Coverage Masks Peak-Hour Demand Risk in Bike-Sharing Forecasts**. No public GitHub URL, Zenodo DOI, or published deposit is claimed yet.

## Files

- `code/run_modern.py`: original modern Capital Bikeshare modeling pipeline, recovered from the authored notebook.
- `code/analisis_complementarios.py`: post-hoc monthly, daylight-saving-time and hypothetical-capacity diagnostics.
- `code/run_uci.py`: independent historical UCI calendar-only benchmark reproduction.
- `data/hour.csv`: historical UCI dataset with attribution; review redistribution licensing before making repository public.
- `results/`: baseline modern derived hourly observations and held-out predictions, reports, and UCI reproduction table.
- `manuscript/`: updated LaTeX source and referenced figures.

## Environment

Python 3.10+; the pinned versions in `requirements.txt` were used for the recovered modern pipeline. Run `python -m pip install -r requirements.txt`.

## Reproduce UCI

The distributed `data/hour.csv` comes from the UCI Bike Sharing dataset (Fanaee-T, 2013), DOI https://doi.org/10.24432/C5W894. In the project root, run:

`python code/run_uci.py --input data/hour.csv --output outputs_uci`

The historical benchmark uses `cnt`, quantile regressors (0.1/0.9), calendar-only features `hr`, `weekday`, `mnth`, `weekend`, training 2011 (8,645 rows), calibration Jan–Jun 2012 (4,358), holdout Jul–Dec 2012 (4,376). `weekend` identifies Saturday or Sunday in the UCI weekday coding. Rush is 07–09 and 16–19 daily. Do not add contemporary weather to these predictors.

## Reproduce modern model from retained hourly series

`python code/run_modern.py --hourly results/modern_hourly_2024_2025.csv --output outputs_modern`

`python code/compare_reference.py --reference results --reproduced outputs_modern`

`python code/analisis_complementarios.py --predictions outputs_modern/modern_predictions.csv --output outputs_supplemental`

To reconstruct from source archives instead, download the 24 official `YYYYMM-capitalbikeshare-tripdata.zip` archives into `archives/` and run `python code/run_modern.py --archives archives --output outputs_modern`. The original archives are not included. Authors separately reported a successful SHA-256 and hourly-count audit on Windows.

## Reproducibility limitation

The modern aggregation uses *naive local clock hours*. The repeated 01:00 during fall-back is combined; the nonexistent 02:00 during spring-forward is filled at zero in the fixed grid. The exact original aggregation is reproduced, not a timezone-aware canonical hour series. Reported capacity penalties are hypothetical proxies, not realized stockouts.

## Before public release

Confirm license and appropriate attribution for both data sources, remove any private files, check that figure and manuscript content are final, create a GitHub repository, and optionally archive a release in Zenodo to obtain a DOI. After it is genuinely available, replace the repository placeholder in the manuscript's Data and Code Availability section.
