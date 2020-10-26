#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 19:49:24 2020

@author: paulmora
"""

# %% Preliminaries

# Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.api import STLForecast
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA

# Paths
main_path = r"/Users/paulmora/Documents/projects/dengai"
raw_path = r"{}/00 Raw".format(main_path)
code_path = r"{}/01 Code".format(main_path)
data_path = r"{}/02 Data".format(main_path)
output_path = r"{}/03 Output".format(main_path)
approach_method = "stl_arimax"

# Loading the data
preprocessed_data = pd.read_csv(r"{}/preprocessed_data.csv".format(data_path))
data_dict = {}
cities = set(preprocessed_data.loc[:, "city"])
for city in cities:
    bool_city = preprocessed_data.loc[:, "city"] == city
    subset_data = preprocessed_data.loc[bool_city, :].reset_index(drop=True)
    data_dict[city] = subset_data

# Initial feature names
features = [
    'ndvi_ne',
    'ndvi_nw',
    'ndvi_se',
    'ndvi_sw',
    'precipitation_amt_mm',
    'reanalysis_air_temp_k',
    'reanalysis_avg_temp_k',
    'reanalysis_dew_point_temp_k',
    'reanalysis_max_air_temp_k',
    'reanalysis_min_air_temp_k',
    'reanalysis_precip_amt_kg_per_m2',
    'reanalysis_relative_humidity_percent',
    'reanalysis_sat_precip_amt_mm',
    'reanalysis_specific_humidity_g_per_kg',
    'reanalysis_tdtr_k',
    'station_avg_temp_c',
    'station_diur_temp_rng_c',
    'station_max_temp_c',
    'station_min_temp_c',
    'station_precip_mm',
    ]

# %% Functions


def acf_plots(y, max_lags, city_name):
    fig, axs = plt.subplots(figsize=(20, 10))
    sm.graphics.tsa.plot_acf(y.values.squeeze(),
                             lags=max_lags, ax=axs, missing="drop")
    axs.set_title("Autocorrelation Plot", fontsize=18)
    axs.tick_params(axis="both", labelsize=16)
    fig.tight_layout()


def stl_decomposing(y, period, city_name):

    time_series = y.copy()
    res = STL(time_series, period=period, robust=True).fit()
    fig, axs = plt.subplots(2, 1, figsize=(20, 20))
    time_series.plot(ax=axs[0], label="Original Series", color="orange")
    time_series_wo_season = time_series - res.seasonal
    time_series_wo_season.plot(ax=axs[1],
                               label="Original Series - Seasonality")
    for ax in axs.ravel():
        ax.legend(prop={"size": 18})
        ax.tick_params(axis="both", labelsize=16)
    fig.tight_layout()
    return res.resid


def finding_moving_peaks(y, lag, threshold):
    signals = np.zeros(len(y))

    avg_filter = [0] * len(y)
    std_filter = [0] * len(y)

    avg_filter[lag-1] = np.mean(y[0:lag])
    std_filter[lag-1] = np.std(y[0:lag])

    for i in range(lag, len(y)):

        if abs(y[i] - avg_filter[i-1]) > (threshold * std_filter[i-1]):
            signals[i] = 1
        else:
            signals[i] = 0

        avg_filter[i] = np.mean(y[(i-lag+1):i+1])
        std_filter[i] = np.std(y[(i-lag+1):i+1])
    return signals


# %% Forecasting SJ

data = data_dict["sj"]
y = data.dropna().loc[:, "total_cases"]
forecasting_length = sum(data.loc[:, "total_cases"].isna())
year_in_weeks = 52

# Finding frequency
acf_plots(y, year_in_weeks*4, "sj")

# Creating decomposed series
decomposed_series = stl_decomposing(y, year_in_weeks, "sj")

# Finding spikes
signals = finding_moving_peaks(y, 104, 3)

scaled_y_series = y / np.max(y)
fig, axs = plt.subplots(figsize=(20, 10))
axs.plot(scaled_y_series)
axs.plot(signals)

# Correlation matrix with peakes
feature_df = data.dropna().loc[:, features]
feature_df.loc[:, "signal"] = signals
a = feature_df.corr()



# %% Forecasting IQ

data = data_dict["iq"]
y = data.dropna().loc[:, "total_cases"]
forecasting_length = sum(data.loc[:, "total_cases"].isna())

# Finding frequency
acf_plots(y, year_in_weeks*4, "iq")

# Creating decomposed series
decomposed_series = stl_decomposing(y, year_in_weeks, "iq")

# Finding optimal ARMA specification
acf_pacf_plots(decomposed_series, year_in_weeks, "iq")
stlf = STLForecast(y, ARIMA, period=year_in_weeks, robust=True,
                   model_kwargs=dict(order=(1, 0, 5), trend="ct"))
stlf_res = stlf.fit()

iq_forecast = stlf_res.forecast(forecasting_length)
iq_forecast[iq_forecast < 0] = 0
fig, axs = plt.subplots(figsize=(20, 10))
axs.plot(y, label="Actual Data")
axs.plot(iq_forecast, label="Forecast")
axs.tick_params(axis="both", labelsize=16)
axs.legend(prop={"size": 16})
fig.tight_layout()
fig.savefig(r"{}/iq_stl_arimax_forecast.png".format(output_path),
            bbox_inches="tight")

# %% Combining forecasts

columns = ["city", "year", "weekofyear", "total_cases"]
predictions_df = pd.DataFrame(columns=columns)

for city, prediction in zip(["sj", "iq"], [sj_forecast, iq_forecast]):
    data = data_dict[city].copy()
    bool_target_nan = data.loc[:, "total_cases"].isna()
    relevant_data = data.loc[bool_target_nan, columns]
    rounded_preds = round(prediction).astype(int)
    relevant_data.loc[:, "total_cases"] = rounded_preds.values
    predictions_df = predictions_df.append(relevant_data)

predictions_df = predictions_df.astype({"total_cases": int})
predictions_df.to_csv("{}/{}/predictions.csv".format(output_path,
                                                     approach_method),
                      index=False)

# Score: 28.5865