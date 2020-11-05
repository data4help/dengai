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
import seaborn as sns
import scipy.stats as st

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
    fig, axs = plt.subplots(nrows=2, figsize=(20, 10))
    sm.graphics.tsa.plot_acf(y.values.squeeze(),
                             lags=max_lags, ax=axs[1], missing="drop")
    axs[1].set_title("Autocorrelation Plot", fontsize=18)
    axs[0].set_title("Original Time Series", fontsize=18)
    axs[0].plot(y)
    for ax in axs.ravel():
        ax.tick_params(axis="both", labelsize=16)
    fig.tight_layout()
    fig.savefig(r"{}/{}/{}_raw_acf.png".format(output_path,
                                               approach_method,
                                               city_name),
                bbox_inches="tight")


def winsorizer(time_series, level, city_name):

    decimal_level = level / 100
    wind_series = st.mstats.winsorize(time_series, limits=[0, decimal_level])
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))
    axs[0].plot(time_series, label="Original Time Series")
    axs[1].plot(wind_series,
                label="Winsorized at the {}% level".format(level))
    axs = axs.ravel()
    for axes in axs.ravel():
        axes.legend(prop={"size": 16}, loc="upper right")
        axes.tick_params(axis="both", labelsize=16)
    fig.tight_layout()
    fig.savefig(r"{}/{}/{}_win.png".format(output_path,
                                           approach_method,
                                           city_name),
                bbox_inches="tight")
    return wind_series


def stl_decomposing(y, period, city_name):
    time_series = y.copy()
    res = STL(time_series, period=period, robust=True).fit()
    fig, axs = plt.subplots(3, 1, figsize=(30, 20))
    time_series.plot(ax=axs[0], label="Original Series")
    time_series_wo_season = time_series - res.seasonal
    res.seasonal.plot(ax=axs[1])
    time_series_wo_season.plot(ax=axs[2])
    axs[0].set_title("Original Series", fontsize=30)
    axs[1].set_title("Seasonality", fontsize=30)
    axs[2].set_title("Original Series Minus Seasonality", fontsize=30)
    for ax in axs.ravel():
        ax.tick_params(axis="both", labelsize=25)
    fig.tight_layout()
    fig.savefig(r"{}/{}/{}_decomposed_acf.png".format(output_path,
                                                      approach_method,
                                                      city_name),
                bbox_inches="tight")

    return time_series_wo_season


def acf_pacf_plots(time_series, nlags, city_name):

    # Plotting results
    no_nan_time_series = time_series.dropna()
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 10), sharey=True)
    sm.graphics.tsa.plot_acf(no_nan_time_series,
                             lags=nlags, fft=True, ax=axs[0])
    sm.graphics.tsa.plot_pacf(no_nan_time_series,
                              lags=nlags, ax=axs[1])
    for ax, title in zip(axs.ravel(), ["ACF", "PACF"]):
        ax.tick_params(axis="both", labelsize=16)
        ax.set_title(title, fontsize=18)
        ax.set_title(title, fontsize=18)
    fig.tight_layout()
    fig.savefig(r"{}/{}/{}_acf_pacf.png".format(output_path,
                                                approach_method,
                                                city_name),
                bbox_inches="tight")


def box_jenkins_lb_finder(y, p, q, year_in_weeks, city_name):
    lb_df = pd.DataFrame(columns=["MA_{}".format(x) for x in range(1, q)],
                         index=["AR_{}".format(x) for x in range(1, p)])
    for i in range(1, p+1):
        for j in range(1, q+1):
            stlf = STLForecast(y, ARIMA, period=year_in_weeks, robust=True,
                               model_kwargs=dict(order=(i, 0, j), trend="c"))
            stlf_res = stlf.fit()
            results_as_html = stlf_res.summary().tables[2].as_html()
            results_df = pd.read_html(results_as_html, index_col=0)[0]
            lb_df.loc["AR_{}".format(i),
                      "MA_{}".format(j)] = results_df.iloc[0, 0]

    # Create heatmaps out of all dataframes
    fig, axs = plt.subplots(figsize=(20, 10))
    sns.heatmap(lb_df.astype(float), annot=True, fmt=".2f",
                ax=axs, annot_kws={"size": 18},
                vmin=0.05,
                vmax=0.25)
    axs.set_xlabel("MA Terms", fontsize=20)
    axs.set_ylabel("AR Terms", fontsize=20)
    axs.tick_params(axis="both", labelsize=20)
    fig.tight_layout()
    fig.savefig(r"{}/{}/{}_lb_comp.png".format(output_path, approach_method,
                                               city_name),
                bbox_inches="tight")


# %% Forecasting SJ

data = data_dict["sj"]
y = data.dropna().loc[:, "total_cases"]
forecasting_length = sum(data.loc[:, "total_cases"].isna())
year_in_weeks = 52

# Finding frequency
acf_plots(y, year_in_weeks*12, "sj")

# Winsorize
masked_y = winsorizer(y, 2.5, "sj")
win_y = pd.Series(np.ma.getdata(masked_y))

# Creating decomposed series
seasonality = 104
decomposed_series = stl_decomposing(win_y, seasonality, "sj")

# Finding optimal ARMA specification
acf_pacf_plots(decomposed_series, year_in_weeks, "sj")

# Comparing models
p = 6
q = 10
box_jenkins_lb_finder(win_y, p, q, seasonality, "sj")

# Specifying optimal model
stlf = STLForecast(win_y, ARIMA, period=seasonality, robust=True,
                   model_kwargs=dict(order=(3, 0, 1), trend="c"))
stlf_res = stlf.fit()
stlf_res.summary()

# In sample predictions and residual plot
sj_forecast = stlf_res.forecast(forecasting_length)
sj_forecast[sj_forecast < 0] = 0
in_sample = stlf_res.get_prediction(0, len(y),
                                    dynamic=len(y)-260).summary_frame()

fig, axs = plt.subplots(figsize=(20, 10))
axs.scatter(list(range(len(y))), y, label="Actual Data", color="blue")
axs.plot(in_sample.loc[:, "mean"]-10, label="In sample", color="red")
axs.fill_between(in_sample.index,
                 in_sample.loc[:, "mean_ci_lower"],
                 in_sample.loc[:, "mean_ci_upper"],
                 color="red", alpha=0.1)


axs.plot(sj_forecast, label="Forecast", color="orange")
axs.tick_params(axis="both", labelsize=16)
axs.legend(prop={"size": 16})
fig.tight_layout()
fig.savefig(r"{}/sj_stl_arimax_forecast.png".format(output_path),
            bbox_inches="tight")

predict.predicted_mean.loc['1977-07-01':].plot(ax=ax, style='r--', label='One-step-ahead forecast')
ci = predict_ci.loc['1977-07-01':]
ax.fill_between(ci.index, ci.iloc[:,0], ci.iloc[:,1], color='r', alpha=0.1)
predict_dy.predicted_mean.loc['1977-07-01':].plot(ax=ax, style='g', label='Dynamic forecast (1978)')
ci = predict_dy_ci.loc['1977-07-01':]
ax.fill_between(ci.index, ci.iloc[:,0], ci.iloc[:,1], color='g', alpha=0.1)



# %% Forecasting IQ

data = data_dict["iq"]
y = data.dropna().loc[:, "total_cases"]
forecasting_length = sum(data.loc[:, "total_cases"].isna())
year_in_weeks = 52

# Finding frequency
acf_plots(y, year_in_weeks*8, "iq")

# Creating decomposed series
seasonality = 52
decomposed_series = stl_decomposing(y, seasonality, "iq")

# Finding optimal ARMA specification
acf_pacf_plots(decomposed_series, year_in_weeks, "iq")

# Comparing models
p = 3
q = 9
box_jenkins_lb_finder(y, p, q, seasonality, "iq")

# Specifying optimal model
stlf = STLForecast(y, ARIMA, period=seasonality, robust=True,
                   model_kwargs=dict(order=(1, 0, 1), trend="c"))
stlf_res = stlf.fit()
print(stlf_res.summary())

# In sample predictions and residual plot
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