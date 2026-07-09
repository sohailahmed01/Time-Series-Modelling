# Forecasting Weekly German Electricity Demand

A reproducible time-series study that forecasts German electricity demand and
compares a range of models, from simple statistical benchmarks to machine-learning
and neural approaches. All work lives in a single, run-top-to-bottom notebook.

## Aim

Forecast weekly German electricity demand for a two-year test window and judge
whether more complex models improve on a strong seasonal benchmark, weighing
accuracy against interpretability and complexity.

Research questions:

1. How well do simple benchmark methods forecast weekly demand?
2. Does a SARIMA model improve on the seasonal benchmarks?
3. Do temperature and holiday covariates improve accuracy?
4. Do feature-based or neural models justify their extra complexity?
5. Which model is most appropriate for an operational setting?

## Data

- **Load:** hourly German load (`DE_load_actual_entsoe_transparency`) from
  Open Power System Data, kept from 1 January 2015 to the end of the file
  (October 2020), aggregated to weekly and daily averages and converted MW to GW.
- **Temperature:** daily mean temperature for Berlin from the Open-Meteo archive
  API, aggregated to weekly mean/min/max plus heating- and cooling-degree
  features. Temperature is an external covariate; because the test window uses
  observed temperature, the SARIMAX forecast is described as conditional.
- **Holidays:** German public holidays, counted per week (a valid future
  covariate, known in advance).

The main target is `load_gw` (weekly mean load in GW).

## How to run

The notebook is written for Google Colab (a GPU runtime is recommended for the
LSTM section).

1. Open `electricity_demand_forecasting.ipynb` in Colab.
2. Provide the load data one of two ways:
   - upload `time_series_60min_singleindex.csv` into the Colab session, or
   - do nothing and let the first cell download it from Open Power System Data.
3. Run the sections in order (Runtime -> Run all).

The SARIMA grid search (Section 3) and the recursive LSTM (Section 6) are the two
slow steps; the full brief-compliant SARIMA grid takes roughly 30-45 minutes.

## Notebook structure

The notebook is organised section by section, mapping to the assignment parts:

| Section | Content |
|---|---|
| 0 | Setup and plotting style |
| 1 | Data acquisition, EDA, seasonality, stationarity (ADF, ACF, PACF, differencing) |
| 2 | Benchmark models (mean, naive, seasonal naive, drift) over a 104-week horizon |
| 3 | SARIMA: AIC grid search, residual diagnostics, forecast with confidence intervals |
| 4 | SARIMAX with temperature (conditional forecast) |
| 5 | Feature-based machine learning (random forest, gradient boosting) |
| 6 | LSTM on the hourly series (tuned; recursive multi-step forecast) |
| 7 | Analysis questions, supported by an evidence table |
| 8 | Consolidated metric table, master forecast plot, regime diagnostics |

## Models

- **Benchmarks:** mean, naive, seasonal naive (period 52), drift.
- **SARIMA:** orders searched by AIC over p in [0,6], d in [0,2], q in [0,6] with a
  seasonal part of (1,1,1) at period 52.
- **SARIMAX:** SARIMA plus temperature and degree-day regressors.
- **Feature ML:** supervised table of calendar, holiday, temperature, lag and
  rolling-window features; lag and rolling features are shifted so they use past
  information only, and the test horizon is produced recursively.
- **LSTM:** trained on hourly data with train-only scaling and a one-week
  look-back; forecast recursively over the two-year horizon.

## Evaluation

The series is split in time order into 195 training weeks and the final 104 test
weeks (no shuffling). Every model is scored on the same 104-week test window with MAE, RMSE, MASE
(scaled by the in-sample seasonal-naive error) and Bias, and each is compared
against the seasonal naive. Additional diagnostics cover the Christmas and New
Year weeks and the hottest and coldest weeks.

## Avoiding data leakage

- Lag and rolling-window features are shifted before use, so they never contain
  future values.
- Scaling for the LSTM is fitted on the training portion only.
- The two-year forecasts are produced recursively, so no future actual enters the
  horizon.
- The train/test split is the final 104 weeks in time order (no shuffling).
- Observed temperature is used in the test window, so the SARIMAX result is
  reported as a conditional forecast.

## Outputs

Running the notebook writes to an `outputs/` folder next to it:

```
outputs/metrics/model_comparison.csv     # MAE, RMSE, MASE, Bias per model
outputs/forecasts/all_forecasts.csv      # actual + each model's forecast
outputs/figures/forecast_comparison.png  # master comparison plot
```

## Requirements

See `requirements.txt`. On Colab, only `holidays` needs installing (the notebook
does this); everything else is preinstalled.

## Files

```
electricity_demand_forecasting.ipynb   # the analysis
requirements.txt                       # dependencies
time_series_60min_singleindex.csv      # load data (optional local copy)
```
