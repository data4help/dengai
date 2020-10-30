#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 10:44:08 2020

@author: paulmora
"""

# %% Preliminaries

# Packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
import scipy.stats as st
from tbats import TBATS
import copy

from statsmodels.tsa.stattools import grangercausalitytests
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

# Paths
main_path = r"/Users/paulmora/Documents/projects/dengai"
raw_path = r"{}/00 Raw".format(main_path)
code_path = r"{}/01 Code".format(main_path)
data_path = r"{}/02 Data".format(main_path)
output_path = r"{}/03 Output".format(main_path)
approach_method = "deasonalized_boosting"

# Loading the data
preprocessed_data = pd.read_csv(r"{}/preprocessed_data.csv".format(data_path))
data_dict = {}
cities = set(preprocessed_data.loc[:, "city"])
for city in cities:
    bool_city = preprocessed_data.loc[:, "city"] == city
    subset_data = preprocessed_data.loc[bool_city, :].reset_index(drop=True)
    data_dict[city] = subset_data

# %% Configurations

#%%% Lasso model
lasso_model_dict = {
    # Lasso Regression
    "model": linear_model.Lasso(random_state=28, fit_intercept=True),
    "param": {
        "alpha": [1e-8, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
        }
    }

gbr_model_dict = {
    # Gradient Boosting Regressor
        "model": GradientBoostingRegressor(random_state=28),
        "param": {
            "n_estimators": [1_000],
            "learning_rate": [0.1, 0.2, 0.3],
            "min_samples_split": [2, 5, 10, 15, 100],
            "min_samples_leaf": [1, 2, 5, 10]
        }
    }

#%%% Variable Names
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
    'station_precip_mm']

# %% Functions


def acf_plots(y, max_lags, city_name):
    fig, axs = plt.subplots(figsize=(20, 10))
    sm.graphics.tsa.plot_acf(y.values.squeeze(),
                             lags=max_lags, ax=axs, missing="drop")
    axs.set_title("Autocorrelation Plot", fontsize=18)
    axs.tick_params(axis="both", labelsize=16)
    fig.tight_layout()
    fig.savefig(r"{}/{}/{}_raw_acf.png".format(output_path,
                                               approach_method,
                                               city_name),
                bbox_inches="tight")


def tbats_deseasonalizing(y, periods, city_name):

    time_series = y.copy()
    estimator = TBATS(seasonal_periods=periods)
    tbats_model = estimator.fit(time_series)
    in_sample_seasonality = tbats_model.forecast(len(time_series))

    fig, axs = plt.subplots(nrows=2, figsize=(20, 15), sharex=True)
    axs[0].plot(time_series, label="Actual Time Series")
    axs[0].plot(in_sample_seasonality, label="TBATS Seasonality")

    deseasonalized_series = time_series - in_sample_seasonality
    axs[1].plot(deseasonalized_series, label="Deseasonalized Series")

    for ax in axs.ravel():
        ax.legend(prop={"size": 18})
        ax.tick_params(axis="both", labelsize=16)

    fig.tight_layout()
    fig.savefig(r"{}/{}/{}_raw_acf.png".format(output_path,
                                               approach_method,
                                               city_name),
                bbox_inches="tight")

    return deseasonalized_series, in_sample_seasonality


def granger_causality_finder(y, data, features, maxlag):

    lag_df = pd.DataFrame(data=0, columns=["sign_lags"], index=features)
    for col in features:
        gc_data = pd.DataFrame(y)
        gc_data.loc[:, col] = data.loc[:, col]

        gc_res = grangercausalitytests(gc_data, maxlag=maxlag, verbose=False)
        lag_results = list(gc_res.values())

        for num, pvalue in enumerate(lag_results, start=1):
            if pvalue[0]["ssr_ftest"][1] < 0.05:
                lag_df.loc[col, "sign_lags"] = num

    lag_df.reset_index(inplace=True)
    fig, axs = plt.subplots(figsize=(20, 10))
    sns.barplot(x="index", y="sign_lags", data=lag_df, ax=axs)
    axs.tick_params(axis="both", labelsize=16)
    axs.tick_params(axis="x", rotation=90)
    axs.set_xlabel("Features", fontsize=18)
    axs.set_ylabel("Relevant Lags", fontsize=18)
    fig.tight_layout()

    lag_df.index = lag_df.loc[:, "index"]
    lag_df.drop(columns=["index"], inplace=True)

    return lag_df


def lag_creation(y, data, features, lag_df):
    lagged_data = data.loc[:, features]
    lagged_data.loc[:, "total_cases"] = y
    for col in features:
        num_lags = lag_df.loc[col, "sign_lags"]
        if num_lags != 0:
            for lag in range(1, num_lags+1):
                lag_col = lagged_data.loc[:, col].shift(lag)
                lagged_data.loc[:, "{}_{}".format(col, str(lag))] = lag_col
    return lagged_data


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
    fig.savefig(r"{}/{}_win.png".format(output_path, city_name),
                bbox_inches="tight")
    return wind_series


def model_train(model_set, X, y, scoring, time_series=False,
                standard_scaled=False, minmax_scaled=False):
    """
    This function trains the model for interpolating the columns which have
    more than one percent missing data

    Parameters
    ----------
    model_set : dict
        This variable contains the model as well as the parameters which
        should be used for the gridsearch
    X : array
        numpy array with the non-missing feature rows from the target
    y : array
        numpy array with the non-missing target rows

    Returns
    -------
    model_info : dict
        gives out a dictionary with the performance of the model as well as
        the model itself.
    """

    model = copy.deepcopy(model_set["model"])
    parameters = model_set["param"]

    if standard_scaled:
        X = StandardScaler().fit_transform(X)

    if minmax_scaled:
        X = MinMaxScaler().fit_transform(X)

    model_info = {}
    if time_series:
        tscv = TimeSeriesSplit(n_splits=5)
        clf = GridSearchCV(model, parameters, n_jobs=-1, cv=tscv,
                           scoring=scoring)
    else:
        clf = GridSearchCV(model, parameters, n_jobs=-1,
                           scoring=scoring)
    clf.fit(X, y)
    scores = [(clf.cv_results_["mean_test_score"]).max()]
    model_info["scores"] = scores

    best_params = clf.best_params_
    for key, value in best_params.items():
        setattr(model, key, value)
    model.fit(X, y)
    model_info["model"] = model
    return model_info


def lasso_column_finder(y_name, data, features):

    non_nan_data = data.dropna()
    y = non_nan_data.loc[:, y_name]
    X = non_nan_data.drop(columns=["total_cases", y_name])

    lasso_model_info = model_train(**{"model_set": lasso_model_dict,
                                      "X": X, "y": y,
                                      "time_series": True,
                                      "standard_scaled": True,
                                      "minmax_scaled": False,
                                      "scoring": "neg_mean_absolute_error"})
    bool_non_null = lasso_model_info["model"].coef_ != 0
    lasso_chosen_columns = list(X.columns[bool_non_null])
    return lasso_chosen_columns


# %% Forecasting SJ

data = data_dict["sj"]
y = data.dropna().loc[:, "total_cases"]
forecasting_length = sum(data.loc[:, "total_cases"].isna())
year_in_weeks = 52

acf_plots(y, year_in_weeks*4, "sj")
deseasonalized_series, seasaonlity = tbats_deseasonalizing(y, [year_in_weeks],
                                                           "sj")
granger_df = granger_causality_finder(deseasonalized_series, data,
                                      features, year_in_weeks)
lagged_df = lag_creation(deseasonalized_series, data.copy(),
                         features, granger_df)

lagged_df.loc[:, "deseasonalized_series"] = deseasonalized_series
lasso_columns = lasso_column_finder("deseasonalized_series",
                                    lagged_df, features)

"""
Implement model training and potentially windsorize the data given
the
"""


no_nan_data = 
X_data = lagged_df
lasso_model_info = model_train(**{"model_set": gbr_model_dict,
                                  "X": X, "y": y,
                                  "time_series": True,
                                  "standard_scaled": True,
                                  "minmax_scaled": False,
                                  "scoring": "neg_mean_absolute_error"})

# # %% Forecasting IQ

# data = data_dict["iq"]
# y = data.dropna().loc[:, "total_cases"]
# forecasting_length = sum(data.loc[:, "total_cases"].isna())


# # %% Combining forecasts

# columns = ["city", "year", "weekofyear", "total_cases"]
# predictions_df = pd.DataFrame(columns=columns)

# for city, prediction in zip(["sj", "iq"], [sj_forecast, iq_forecast]):
#     data = data_dict[city].copy()
#     bool_target_nan = data.loc[:, "total_cases"].isna()
#     relevant_data = data.loc[bool_target_nan, columns]
#     rounded_preds = round(prediction).astype(int)
#     relevant_data.loc[:, "total_cases"] = rounded_preds.values
#     predictions_df = predictions_df.append(relevant_data)

# predictions_df = predictions_df.astype({"total_cases": int})
# predictions_df.to_csv("{}/{}/predictions.csv".format(output_path,
#                                                      approach_method),
#                       index=False)

# # Score: 28.5865




# lagged_data = data.loc[:, features]
# lagged_data.loc[:, "total_cases"] = y
# for col in features:
#     num_lags = lag_df.loc[col, "sign_lags"]
#     if num_lags != 0:
#         for lag in range(1, num_lags+1):
#             lag_col = lagged_data.loc[:, col].shift(lag)
#             lagged_data.loc[:, "{}_{}".format(col, str(lag))] = lag_col
