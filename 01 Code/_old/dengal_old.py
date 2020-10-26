#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 00:09:27 2020

@author: paulmora
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 23:29:39 2020

@author: PaulM
"""

def model_build(data, col, cols, missing_rows):
    
    # Checking for valid features
    for col in cols:
        bool_col_nan = data.loc[:, col].isna()
        bool_usable_columns = (~data.loc[bool_col_nan, cols].isna().any()).any()
        print(bool_usable_columns)
    if bool_usable_columns:
        no_nan_target_data = data.loc[:, cols]\
            .dropna(subset=[col])\
            .reset_index(drop=True)
        no_nan_columns = np.array(cols)[~no_nan_target_data.isna().any()]
        cols_wo_col = list(set(no_nan_columns) - set(col))
        
        # Separate train and test data
        test =  no_nan_target_data.loc[missing_rows, :]
        train = no_nan_target_data.drop(missing_rows)
        feature_values = train.loc[:, cols_wo_col].values
        target_values = train.loc[:, col].values
        scaler = StandardScaler().fit(feature_values)
        scaled_train = scaler.transform(feature_values)    
        
        # Predicting missing values
        rfr = RandomForestRegressor(random_state=28, n_estimators=1_000)
        fitted_model = rfr.fit(scaled_train, target_values)
        scaled_test = scaler.transform(test.loc[:, cols_wo_col])
        pred = fitted_model.predict(scaled_test)
        no_nan_target_data.loc[missing_rows, col] = pred
        filled_pred = no_nan_target_data.loc[:, col]
        
        # Already predicting the other values
        bool_target_nan = data.loc[:, col].isna()
        full_features = data.loc[bool_target_nan, cols_wo_col]
        
        return filled_pred

    else:
        return np.nan

def season_trend(data, cols, spike_dict, city_name):

    fig, axs = plt.subplots(nrows=4, ncols=5, sharex=True,
                            figsize=(20, 20))
    plt.subplots_adjust(right=None)
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor="none", top=False, bottom=False,
                    left=False, right=False)
    plt.grid(False)
    plt.xlabel("Time", fontsize=22, labelpad=20)
    plt.ylabel("Values", fontsize=22, labelpad=50)

    axs = axs.ravel()
    for i, col in enumerate(cols):
        period = spike_dict[col]
        time_series = data.loc[:, col]
        axs[i].plot(time_series, color="r")
        if not np.isnan(period):
            res = STL(time_series, period=int(spike_dict[col])+1).fit()
            time_series = time_series - res.trend - res.seasonal
        else:
            res = mk.original_test(time_series)
            if res.trend != "no trend":
                trend_line = [(res.intercept + res.slope * x)
                              for x in range(len(time_series))]
                time_series = time_series - trend_line
        
        axs[i].plot(time_series)
        axs[i].set_title(col, fontsize=12)
        axs[i].tick_params(axis="both", labelsize=16)
        data.loc[:, col] = time_series.copy()

    fig.tight_layout()
    fig.savefig(r"{}/{}_pre_post_season_trend.png".format(output_path,
                                                          city_name),
                bbox_inches="tight")
    return data

# %% Preliminaries

# Packages
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.seasonal import STL
import statsmodels.api as sm

import scipy.stats as st
from scipy import fft
from scipy import signal as sig
from sklearn.preprocessing import StandardScaler

# Paths
main_path = r"/Users/paulmora/Documents/projects/dengal"
raw_path = r"{}/00 Raw".format(main_path)
code_path = r"{}/01 Code".format(main_path)
data_path = r"{}/02 Data".format(main_path)
output_path = r"{}/03 Output".format(main_path)

os.chdir(code_path)
from _config import variable_dict, model_dict
from _functions import model_train, missing_divider

# Loading the data
train_values = pd.read_csv(r"{}/dengue_features_train.csv".format(raw_path))
train_labels = pd.read_csv(r"{}/dengue_labels_train.csv".format(raw_path))
test_values = pd.read_csv(r"{}/dengue_features_test.csv".format(raw_path))

# %% Separating the data for easier access

data = {}
for city in ["sj", "iq"]:
    bool_city_train = train_values.loc[:, "city"] == city
    bool_city_train_labels = train_labels.loc[:, "city"] == city
    bool_city_test = test_values.loc[:, "city"] == city

    train_values_city = train_values.loc[bool_city_train, :]
    train_labels_city = train_labels.loc[bool_city_train_labels, :]
    test_values_city = test_values.loc[bool_city_test, :]

    data.update({"train_values_{}".format(city): train_values_city})
    data.update({"train_labels_{}".format(city): train_labels_city})
    data.update({"test_values_{}".format(city): test_values_city})

# %% Stationarity Tests

"""
Before moving on to any kind of analysis, we have to make sure that all
features do not exhibit any form trend, meaning whether they are stationary.
A trend in the feature could bring something called spurious correlation
which would impact our estimation performance. In order to work-around
that problem, we check first whether we find a significant linear trend,
namely a determinstic trend. If we find one, we remove it, by simply
subtracting the linear fit.

Lastly we have to check whether any time series suffers from a stochastic
trend. This is tested through an augmented dickey fuller test. Normally a
KPSS test would have been necessary, but since we already removed any kind of
deterministic trend, using an ADF is just fine
"""

df_data = data["train_values_sj"].copy()
all_cols = variable_dict["all_variables"]
var = ["coefficient", "tvalue", "d_trend", "adf", "s_trend"]
df_stationary = pd.DataFrame(data=np.nan, index=all_cols, columns=var)
dict_trend = {}
time = np.arange(1, len(df_data)+1)
critical_norm = st.norm.ppf(.975)

nrows, ncols = 4, 5
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=True,
                        figsize=(30, 30))
axs = axs.ravel()

for i, col in enumerate(all_cols):
    # estimating the significance of a linear trend
    series = df_data.loc[:, col].copy()
    bool_finite = np.isfinite(series)
    X, y = time[bool_finite], series[bool_finite]
    X = sm.add_constant(X)
    model = sm.OLS(y, X)
    results = model.fit(cov="H2")
    dict_trend[col] = results
    df_stationary.loc[col, "coefficient"] = results.params[1]
    df_stationary.loc[col, "tvalue"] = results.tvalues[1]

    axs[i].plot(time, series)
    axs[i].set_title(col, fontsize=20)
    axs[i].tick_params(axis="both", labelsize=20)

    if abs(df_stationary.loc[col, "tvalue"]) > critical_norm:
        axs[i].plot(time[bool_finite], results.predict(X), color="r",
                    linewidth=3)
        df_stationary.loc[col, "d_trend"] = True
        series = y - results.predict(X)
        df_data.loc[:, col] = series
    else:
        df_stationary.loc[col, "d_trend"] = False

    results = adfuller(series[bool_finite], regression="c", autolag="AIC")
    df_stationary.loc[col, "adf"] = results[0]
    if results[1] < 0.05:
        df_stationary.loc[col, "s_trend"] = False
    else:
        df_stationary.loc[col, "s_trend"] = True
fig.tight_layout()

fig.savefig(r"{}/stationary_checks.png".format(output_path),
            bbox_inches="tight")

# %% Distribution analysis

"""
Given that in the following interpolation section we will apply some
GradientBoosting regressions, we should make sure that our numeric variables
have not massive outliers, which could affect the model performance. For that
reason we plot a density plot for all variables to get an indication about
their distribution.

We will highlight those plots which exhibit a skewness higher than absolute
one - (https://www.researchgate.net/post/What_is_the_acceptable_range_of_skewness_and_kurtosis_for_normal_distribution_of_data_if_sig_value_is_005)
"""

fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 20))
axs = axs.ravel()
all_variables = variable_dict["all_variables"]
for i, col in enumerate(all_variables):
    variable = df_data.loc[:, col].copy()
    bool_non_nan = ~variable.isnull()
    non_nan_variable = variable[bool_non_nan]
    skewness = st.skew(non_nan_variable)
    axs[i].set_title(col, fontsize=15)
    if abs(skewness) > 1:
        if min(non_nan_variable) <= 0:
            axs[i].tick_params(axis="both", labelsize=20)
            sns.distplot(non_nan_variable, color="r", ax=axs[i])
            axs[i].set_xlabel("")
            axs[i] = axs[i].twinx()
            non_nan_variable = np.log((non_nan_variable
                                       + abs(min(non_nan_variable))
                                       + 1))
        # df_data.loc[:, col] = np.log(non_nan_variable)
    sns.distplot(non_nan_variable, color="b", ax=axs[i])
    axs[i].tick_params(axis="both", labelsize=20)
    axs[i].set_xlabel("")
fig.tight_layout()

fig.savefig(r"{}/skewness.png".format(output_path),
            bbox_inches="tight")

# %% Interpolation - First Analysis

"""
One of the first things to consider is to see how many missing observations
we are dealing with, for that we are starting with plotting the missing
observations.
"""

# Plotting the problem
fig = msno.matrix(df_data, figsize=(20, 10))
fig.figure.savefig(r"{}/missing_data.png".format(output_path),
                   bbox_inches="tight")

"""
From the plot before we can see that we have the amount of missing data
differs, substantially throughout the different time series. Next it would
be good to know how much of each time series is missing percentage wise. We
do that for both, training as well as test data.
"""

percentage_missing = pd.DataFrame(columns=["Variable", "Missing", "type"])
i = 0
for data_type in ["train", "test"]:
    df_type = data["{}_values_sj".format(data_type)].copy()
    for col in all_cols:
        bool_nan = df_type.loc[:, col].isnull()
        pct_nan = (sum(bool_nan) / len(bool_nan)) * 1_00
        percentage_missing.loc[i, "Variable"] = col
        percentage_missing.loc[i, "Missing"] = pct_nan
        percentage_missing.loc[i, "type"] = data_type
        i += 1

fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))
sns.barplot(data=percentage_missing, x="Variable", y="Missing",
            hue="type", ax=axs)
axs.tick_params(axis="both", labelsize=16, labelrotation=90)
axs.set_ylabel("Percentage of missing observations", fontsize=18)
axs.set_xlabel("Variables", fontsize=18)
axs.axhline(y=1, xmin=0, xmax=1, linewidth=3, color="r", linestyle="dashed",
            label="1 Percent Hurdle")
axs.legend(prop={"size": 20})
fig.savefig(r"{}/missing_data_pct.png".format(output_path),
            bbox_inches="tight")

# %% Interpolation - Taking actions

"""
We start by filling all columns which have less than 1 percent missing by
using a cubic interpolation method. To do so we first identify which columns
are meant and then loop over them
"""

bool_smaller_one = percentage_missing.loc[:, "Missing"] < 1
bool_train = percentage_missing.loc[:, "type"] == "train"
col_subset = list(percentage_missing.loc[bool_smaller_one & bool_train,
                                         "Variable"])
for col in col_subset:
    df_data.loc[:, col].interpolate(method="cubic", inplace=True)

"""
Now we quickly make sure that this method actually filled up the columns
and that there is no more missing data anymore
"""

fig = msno.matrix(df_data, figsize=(20, 10))
fig.figure.savefig(r"{}/missing_data_initial.png".format(output_path),
                   bbox_inches="tight")

"""
As we can see from the picture above, all columns with less than one percent
of their rows filled are filled up entirely. We will now use all filled
columns to build a prediction model for each variable with missing data.
The model building procedure starts with the least populated model
"""

target_cols = list(percentage_missing.loc[~bool_smaller_one & bool_train,
                                          "Variable"])
model = model_dict["xgb"]
filling_models = {}
for col in tqdm(reversed(target_cols)):
    filling_models[col] = {}

    y_df, X_df = df_data.loc[:, col].copy(), df_data.loc[:, col_subset].copy()
    X_non_nan, X_nan, y_non_nan = missing_divider(X_df, y_df)
    model_set = model_train(model_dict["xgb"], X_non_nan,
                            y_non_nan, time_series=False)
    y_pred = model_set["model"].predict(X_nan)
    y_df[y_df.isnull()] = y_pred.copy()
    df_data.loc[:, col] = y_df.copy()
    col_subset.append(col)
    filling_models[col].update(model_set)

"""
In order to see how well our interpolation model works we plot the
normalized root-mean-square deviation of each variable.
https://en.wikipedia.org/wiki/Root-mean-square_deviation
"""

df_filling = pd.DataFrame(columns=["variable", "NRMSD"],
                          index=list(range(len(target_cols))))
for i, col in enumerate(target_cols):
    df_filling.loc[i, "variable"] = col
    rmsd = np.sqrt(abs(filling_models[col]["scores"][0]))
    minimum, maximum = min(df_data.loc[:, col]), max(df_data.loc[:, col])
    df_filling.loc[i, "NRMSD"] = rmsd/(maximum-minimum)
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))
axs = sns.barplot(x="variable", y="NRMSD", data=df_filling)
axs.tick_params(axis="both", labelsize=16)
axs.set_ylabel("Normalized root-mean-square deviation", fontsize=18)
axs.set_xlabel("Interpolated Variables", fontsize=18)
fig.savefig(r"{}/interpolation_filling.png".format(output_path),
            bbox_inches="tight")

# %% Seasonality

"""
First we find the powerspectrum of each variable. A power-spectrum gives us
an idea about which frequency drives the overall series. If for example the
52 week frequency drives the series overly proportionally, then it is likely
that the series suffers from yearly seasonality.

In order to test for that, we start by calculating the fourier transform of
each series. Afterwards we plot a power-spectrum. A driving frequency is
defined as such if it exceeds ten times the median. This is an arbitrary
threshold, but a decision had to be made.
"""

fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 20),
                        sharex=True)
plt.subplots_adjust(right=None)
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor="none", top=False, bottom=False,
                left=False, right=False)
plt.grid(False)
plt.xlabel("Frequency [1 / Hour]", fontsize=22, labelpad=20)
plt.ylabel("Amplitude", fontsize=22, labelpad=50)
spikes = {}
axs = axs.ravel()
for i, col in enumerate(all_variables):

    signal = df_data.loc[:, col].copy()
    fft_output = fft.fft(signal.values)
    power = np.abs(fft_output)
    freq = fft.fftfreq(len(signal))

    mask = freq >= 0
    freq = freq[mask]
    power = power[mask]

    mask = (freq > 0) & (freq <= 0.25)
    axs[i].plot(freq[mask], power[mask])
    axs[i].tick_params(axis="both", labelsize=16)
    axs[i].set_title(col, fontsize=12)

    threshold = np.median(power) * 10
    peaks = sig.find_peaks(power[(freq >= 0) & (freq <= 0.25)],
                           prominence=threshold)[0]
    peak_freq = freq[peaks]
    peak_power = power[peaks]
    axs[i].plot(peak_freq, peak_power, 'ro')
    spikes[col] = (1 / peak_freq).tolist()

fig.tight_layout()
fig.savefig(r"{}/fourier_transform.png".format(output_path),
            bbox_inches="tight")

"""
In order to cross-check our findings we plot ACFs for every variable and
indicate at which points the seasonality should occur, according to our results
"""

fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True,
                        figsize=(20, 20))
plt.subplots_adjust(right=None)
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor="none", top=False, bottom=False,
                left=False, right=False)
plt.grid(False)
plt.xlabel("Lags", fontsize=22, labelpad=20)
plt.ylabel("Correlation", fontsize=22, labelpad=50)
axs = axs.ravel()
max_lags = round(max(max(spikes.values())))
for i, col in enumerate(all_variables):
    # Plotting the lines
    series = df_data.loc[:, col].copy()
    sm.graphics.tsa.plot_acf(series.values.squeeze(),
                             lags=max_lags, ax=axs[i], missing="drop")
    axs[i].set_title(col, fontsize=12)
    axs[i].tick_params(axis="both", labelsize=16)

    if len(spikes[col]) != 0:
        for season in spikes[col]:
            multipler = [i for i in range(1, max_lags+1) if i % season == 0]
            for line in multipler:
                axs[i].axvline(line, -1, 1, color="red",
                               label="Periodicity: {}".format(line))
        axs[i].legend(loc="upper center", prop={'size': 16})
fig.tight_layout()
fig.savefig(r"{}/autocorrelation_function.png".format(output_path),
            bbox_inches="tight")

"""
Now we need to find whether the season is multiplicative or additive
we do that in the following manner. We will once subtract and once divide
by the seasonality found through the STL package. We then calculate the
auto-correlation of the residuals in order to find which was the better fit.
Source: https://www.r-bloggers.com/is-my-time-series-additive-or-multiplicative/
"""

fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=True,
                        figsize=(20, 20))
plt.subplots_adjust(right=None)
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor="none", top=False, bottom=False,
                left=False, right=False)
plt.grid(False)
plt.xlabel("Time", fontsize=22, labelpad=20)
plt.ylabel("Values", fontsize=22, labelpad=50)

axs = axs.ravel()
for i, col in enumerate(all_variables):
    if len(spikes[col]) != 0:
        time_series = df_data.loc[:, col]
        res = STL(time_series, period=int(spikes[col][0])).fit()
        cleaned = time_series - res.trend - res.seasonal
        axs[i].plot(df_data.loc[:, col], color="r")
        axs[i].plot(cleaned)
        df_data.loc[:, col] = cleaned
    else:
        axs[i].plot(df_data.loc[:, col])
    axs[i].set_title(col, fontsize=12)
    axs[i].tick_params(axis="both", labelsize=16)

fig.tight_layout()
fig.savefig(r"{}/detrended_deseasonalized_comparison.png".format(output_path),
            bbox_inches="tight")

# %% Potential Non-Linear Variables

"""
Given that the year and the weekofyear potentially be very important
variables, which have a non-linear effect, we apply mean encoding.
Assuming that for example that we have low cases in January and December,
but high cases in June, the variable is not able to convey that correctly
in a linear fashion - we therefore apply mean encoding.
"""

labels = data["train_labels_sj"].copy()
df_data.loc[:, "total_cases"] = labels.loc[:, "total_cases"]

weekmap = df_data.groupby(["weekofyear"])["total_cases"].mean().to_dict()
df_data.loc[:, "weekofyear_enc"] = df_data.loc[:, "weekofyear"].map(weekmap)

# %% Dependent variable

"""
First we check whether we find stationarity of the time series. If not we
have to apply differences first. We test for stationarity by applying a
dicker-fuller test
"""

adf_y = adfuller(labels.loc[:, "total_cases"], regression="c", autolag="AIC")

"""

"""

pacf_data = sm.graphics.tsa.pacf(labels.loc[:, "total_cases"], alpha=0.05)
for num_y, (pacf_value, pacf_conf) in enumerate(zip(pacf_data[0],
                                                    pacf_data[1])):
    lower_conf = pacf_conf[0] - pacf_value
    upper_conf = pacf_conf[1] - pacf_value
    if lower_conf < pacf_value < upper_conf:
        break

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
sm.graphics.tsa.plot_acf(labels.loc[:, "total_cases"], lags=52, ax=axs[0])
sm.graphics.tsa.plot_pacf(labels.loc[:, "total_cases"], lags=52, ax=axs[1])
for ax, title in zip(axs.ravel(), ["ACF", "PACF"]):
    ax.tick_params(axis="both", labelsize=16)
    ax.set_title(title, fontsize=18)
axs[1].axvline(num_y, color="r", linestyle="dashed")
fig.tight_layout()
fig.savefig(r"{}/dependent_pacf_acf.png".format(output_path),
            bbox_inches="tight")

"""
Now it is our job to find out which variables have a significant impact to
the forecasting. We first start finding how many lags are useful we apply
a Granger Causality
"""

maxlag = 52
lag_dict = {}
for col in all_variables:
    data = labels.loc[:, ["total_cases"]]
    data.loc[:, col] = df_data.loc[:, col]

    gc_res = grangercausalitytests(data, maxlag=maxlag, verbose=False)
    lag_results = list(gc_res.values())

    lag_dict[col] = 0
    for num, pvalue in enumerate(lag_results, start=1):
        if pvalue[0]["ssr_ftest"][1] < 0.05:
            lag_dict[col] = num

"""
We now create the lagged values and run a lasso regression
"""

lagged_data = df_data.loc[:, all_variables]
lagged_data.loc[:, "total_cases"] = labels.loc[:, "total_cases"]
for col in all_variables:
    if lag_dict[col] != 0:
        for lag in range(1, lag_dict[col]+1):
            lag_col = lagged_data.loc[:, col].shift(lag)
            lagged_data.loc[:, "{}_{}".format(col, str(lag))] = lag_col

add_variables = ["year", "weekofyear_enc"]
lagged_data.loc[:, add_variables] = df_data.loc[:, add_variables]
lagged_data.loc[:, "total_cases"] = labels.loc[:, "total_cases"]
no_nan_lagged = lagged_data.dropna()

y = no_nan_lagged.loc[:, "total_cases"]
lasso_data = no_nan_lagged.drop(columns=["total_cases"])
X_stand = StandardScaler().fit_transform(lasso_data)

lasso_model = model_train(model_dict["lasso"], X_stand, y, time_series=True)
bool_non_null = lasso_model["model"].coef_ != 0
rel_columns = lasso_data.columns[bool_non_null]

"""
Before we apply the lasso, it would be interesting to see a correlation
heatmap in order to make first hypothesis about the result
"""

total_case_corr = abs(no_nan_lagged.corr().loc[:, ["total_cases"]])
sorted_corr = total_case_corr.sort_values("total_cases", ascending=False)[:30]

fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 20))
sns.heatmap(sorted_corr, annot=True, ax=axs)

"""
After figuring out which variables could be of any importance, we then create
the lags of the target variable and add those columns
"""

y_columns = ["total_cases_{}".format(x) for x in np.arange(1, num_y+1)]
cols = list(rel_columns) + y_columns
max_lag = max([x.split("_")[-1] for x in cols if x.split("_")[-1].isnumeric()])

for col in cols:
    for lag in range(1, int(max_lag)+1):
        lag_col = lagged_data.loc[:, col].shift(lag)
        lagged_data.loc[:, "{}_{}".format(col, str(lag))] = lag_col

cols_w_target = cols + ["total_cases"]
final_data = lagged_data.loc[:, cols_w_target].dropna()

# %% Model building

y = final_data.loc[:, "total_cases"]
X = StandardScaler().fit_transform(final_data)
xgb_model = model_train(model_dict["xgb"], X, y, time_series=True)


# Show error term analysis
res.plot_diagnostics(figsize=(20, 20))

# Forecasting plot

thresh = 300
predict = res.get_prediction(exog=scaled_train_exog[thresh:, :])
predict_ci = predict.conf_int()

predict_dy = res.get_prediction(dynamic=thresh,
                                    exog=scaled_train_exog[thresh:])
predict_dy_ci = predict_dy.conf_int()

fig, ax = plt.subplots(figsize=(20,10))

endog[thresh:].plot(ax=ax, style='o', label='Observed')

# Plot predictions
predict.predicted_mean.loc[thresh:].plot(ax=ax,
                                         style='r--',
                                         label='One-step-ahead forecast')
ci = predict_ci.loc[thresh:]
ax.fill_between(ci.index, ci.iloc[:,0], ci.iloc[:,1], color='r', alpha=0.1)
predict_dy.predicted_mean.loc[thresh:].plot(ax=ax, style='g')
ci = predict_dy_ci.loc[thresh:]
ax.fill_between(ci.index, ci.iloc[:,0], ci.iloc[:,1], color='g', alpha=0.1)

from sklearn.metrics import mean_absolute_error
y_true = data.loc[thresh:, "total_cases"].dropna()
y_pred = round(predict_dy.predicted_mean[thresh:]).astype(int)
mean_absolute_error(y_true, y_pred)

# Actual forecast


res.bic

arma_param = arma_parameterization(y_time_series, 10, 5, "sj")





fit_res = model.fit(disp=False, maxiter=250)
print(res.summary())


res = fit_res.resid
fig,ax = plt.subplots(2,1,figsize=(15,8))
fig = sm.graphics.tsa.plot_acf(res, lags=50, ax=ax[0])
fig = sm.graphics.tsa.plot_pacf(res, lags=50, ax=ax[1])
plt.show()


thresh = 600
predict = fit_res.get_prediction(exog=train_exog.loc[thresh:, :])
predict_ci = predict.conf_int()

predict_dy = fit_res.get_prediction(dynamic=thresh,
                                    exog=train_exog.loc[thresh:, :])
predict_dy_ci = predict_dy.conf_int()




# %% 

def relevant_auto_lags(data, city_name):

    # Prepare data
    target = data.loc[:, "total_cases"].copy()
    no_nan_target = target.dropna()
    forecasting_periods = sum(target.isna())
    min_nlags = forecasting_periods
    max_nlags = min_nlags + 52

    # Find significant lags
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 10), sharey=True)
    pacf_data = sm.graphics.tsa.pacf(no_nan_target, alpha=0.05,
                                      nlags=max_nlags)
    acf_data = sm.graphics.tsa.acf(no_nan_target, alpha=0.05,
                                    fft=True, nlags=max_nlags)


    sign_lags = []
    for num_y, (acf_value, acf_conf,
                pacf_value, pacf_conf) in enumerate(zip(acf_data[0],
                                                        acf_data[1],
                                                        pacf_data[0],
                                                        pacf_data[1])):
        if not num_y > min_nlags:
            continue
        pacf_lower_conf = pacf_conf[0] - pacf_value
        pacf_upper_conf = pacf_conf[1] - pacf_value
        pacf_bool = not pacf_lower_conf < pacf_value < pacf_upper_conf

        acf_lower_conf = acf_conf[0] - acf_value
        acf_upper_conf = acf_conf[1] - acf_value
        acf_bool = not acf_lower_conf < acf_value < acf_upper_conf

        if pacf_bool | acf_bool:
            sign_lags.append(num_y)

    # Plotting results
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 10), sharey=True)
    sm.graphics.tsa.plot_acf(no_nan_target,
                              lags=max_nlags, fft=True, ax=axs[0])
    sm.graphics.tsa.plot_pacf(no_nan_target,
                              lags=max_nlags, ax=axs[1])

    for ax, title in zip(axs.ravel(), ["ACF", "PACF"]):
        for line in sign_lags:
            ax.axvline(line, color="red", alpha=0.2)
        ax.axvline(min_nlags, color="black", label="Minimum Lags")
        ax.tick_params(axis="both", labelsize=16)
        ax.set_title(title, fontsize=18)

    red_patch = mpatches.Patch(color="red", alpha=0.2,
                                label="Significant Lags for ACF or PACF")
    black_patch = mpatches.Patch(color="black",
                                  label="Lower bound of possible lags")
    axs[1].legend(handles=[red_patch, black_patch], prop={"size": 16})

    fig.tight_layout()
    fig.savefig(r"{}/{}_target_acf_pacf.png".format(output_path, city_name),
                bbox_inches="tight")
    return sign_lags


def lag_creation(data, cols, sign_y_lags, lag_dict):

    # Creating the lags
    cols_w_target = cols + ["total_cases"]
    rest_columns = list(set(data.columns) - set(cols_w_target))
    lagged_data = data.loc[:, cols_w_target].copy()
    for col in tqdm(cols_w_target):
        if lag_dict[col] != 0:
            for lag in range(1, lag_dict[col]+1):
                lag_col = lagged_data.loc[:, col].shift(lag)
                lagged_data.loc[:, "{}_{}".format(col, str(lag))] = lag_col

    # Picking the relevant target lags
    bool_target_lags = [x.startswith("total_cases_")
                        for x in lagged_data.columns]
    all_target_lags = lagged_data.columns[bool_target_lags]
    relevant_target_lags = ["total_cases_{}".format(x) for x in sign_y_lags]
    deletable_lags = list(set(all_target_lags) - set(relevant_target_lags))
    relevant_lagged_data = lagged_data.drop(columns=deletable_lags)
    relevant_columns = relevant_lagged_data.columns

    # Adding back the variables
    transformed_data = pd.concat([data.loc[:, rest_columns],
                                  relevant_lagged_data], axis=1)

    return transformed_data, relevant_columns

def creating_date_variables(data):
    
    # Month variable
    time_variable = pd.to_datetime(data.loc[:, "week_start_date"],
                                    format="%Y-%m-%d")
    month_variable = time_variable.dt.month
    data.loc[:, "month"] = month_variable

    # Week day variable
    weekdays = [x.strftime("%A") for x in time_variable]
    data.loc[:, "weekday"] = weekdays
    return data

def encoding_variables(data, enc_variables):

    for col in enc_variables:
        enc_data = data.loc[:, [col] + ["total_cases"]]
        enc_map = enc_data.groupby([col])["total_cases"].mean().to_dict()
        mean_enc_data = enc_data.loc[:, col].map(enc_map)
        data.loc[:, col] = mean_enc_data
    return data

def lasso_relevant_columns_finder(data):

    not_needed_columns = ["city", "type", "index", "week_start_date"]
    relevant_columns = list(set(data.columns) - set(not_needed_columns))
    relevant_data = data.loc[:, relevant_columns]

    no_nan_data = relevant_data.dropna()
    X_data = no_nan_data.loc[:, [x for x in relevant_columns
                                  if x != "total_cases"]]
    y = no_nan_data.loc[:, "total_cases"]
    lasso_model_info = model_train(**{"model_set": lasso_model_dict,
                                      "X": X_data, "y": y,
                                      "time_series": True,
                                      "standard_scaled": True,
                                      "minmax_scaled": False,
                                      "scoring": "neg_mean_squared_error"})
    bool_non_null = lasso_model_info["model"].coef_ != 0
    lasso_chosen_columns = list(X_data.columns[bool_non_null])
    return lasso_chosen_columns



def windsorizer(time_series, level, city_name):

    decimal_level = level / 100
    wind_series = st.mstats.winsorize(time_series, limits=[0, decimal_level])
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))
    axs[0].plot(time_series, label="Original Time Series")
    axs[1].plot(wind_series,
                label="Windsorized at the {}% level".format(level))
    axs = axs.ravel()
    for axes in axs.ravel():
        axes.legend(prop={"size": 16}, loc="upper right")
        axes.tick_params(axis="both", labelsize=16)
    fig.tight_layout()
    fig.savefig(r"{}/{}_wind.png".format(output_path, city_name),
                bbox_inches="tight")
    return wind_series

# Get example data
data = data_dict["sj"].copy()

# Imputing the data
imputer = ImputingData(data.copy(), features, imputation_model, 1, "sj")
imputed_data = imputer.imputed_data.copy()

# Transforming the data
transformer = TrendSeasonality(imputed_data, features, "sj")
transformed_data = transformer.transformed_data.copy()

# Finding the significant amount of auto lags
sign_y_lags = relevant_auto_lags(transformed_data, "sj")

# Finding which other variables could be important
granger_dict = granger_causality(transformed_data, features, sign_y_lags)

# Create the lags
(lagged_data,
  relevant_features) = lag_creation(transformed_data, features,
                                    sign_y_lags, granger_dict)

# Create time variables
date_data = creating_date_variables(lagged_data.copy())

# Encoding of variables
enc_variables = ["weekday", "month", "weekofyear"]
enc_data = encoding_variables(date_data.copy(), enc_variables)

# Lasso regression to find significant ones
lasso_columns = lasso_relevant_columns_finder(enc_data.copy())

# Heatmap
correlation_heatmap(enc_data.copy(), "sj")


# Train the model
xgb_model = model_dict["xgb"]
no_nan_data = enc_data.dropna()
final_model = model_train(**{"model_set": xgb_model,
                              "X": no_nan_data.loc[:, lasso_columns],
                              "y": no_nan_data.loc[:, "total_cases"],
                              "time_series": True,
                              "standard_scaled": True,
                              "scoring": "neg_mean_squared_error"})

def correlation_heatmap(data, city_name):

    correlation_matrix = data.corr()
    target_correlation = correlation_matrix.loc[:, ["total_cases"]]
    sorted_corr = target_correlation.sort_values(by="total_cases")

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 20))
    sns.heatmap(pd.concat([sorted_corr.head(n=10),
                           sorted_corr.tail(n=10)]), annot=True, ax=axs)
    axs.tick_params(axis="both", labelsize=16)

    fig.tight_layout()
    fig.savefig(r"{}/{}_heatcorr.png".format(output_path, city_name),
                bbox_inches="tight")



a = transformer.adf_dict

b = pd.DataFrame(columns=a.keys(), index=["Trend", "Slope",
                                          "Intercept", "Post Trend"])
for key, item in a.items():
    b.loc["Trend", key] = item["pre_trend"]
    try:
        b.loc["Slope", key] = item["slope"]
        b.loc["Intercept", key] = item["intercept"]
        b.loc["Post Trend", key] = item["post_trend"]
    except:
        b.loc["Slope", key] = "NA"
        b.loc["Intercept", key] = "NA"
        b.loc["Post Trend", key] = "NA"
        continue

import openpyxl
b.to_csv(r"/Users/paulmora/Desktop/table.csv")
