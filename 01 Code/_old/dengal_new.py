#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 11:54:37 2020

@author: paulmora
"""

# %% Preliminaries

### Packages
import os
import math
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import pickle

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from statsmodels.tsa.stattools import adfuller, kpss
import pymannkendall as mk
from statsmodels.tsa.seasonal import STL
import statsmodels.api as sm
from tbats import TBATS

import scipy.stats as st
from scipy import fft
from scipy import signal as sig
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error


### Paths
main_path = r"/Users/paulmora/Documents/projects/dengal"
raw_path = r"{}/00 Raw".format(main_path)
code_path = r"{}/01 Code".format(main_path)
data_path = r"{}/02 Data".format(main_path)
output_path = r"{}/03 Output".format(main_path)

os.chdir(code_path)
from _config import (features,
                     model_dict, imputation_model,
                     lasso_model_dict)
from _functions import model_train

### Loading the data
train_values = pd.read_csv(r"{}/dengue_features_train.csv".format(raw_path))
train_labels = pd.read_csv(r"{}/dengue_labels_train.csv".format(raw_path))
test_values = pd.read_csv(r"{}/dengue_features_test.csv".format(raw_path))

# %% Separating the data for easier access

cities = list(set(train_values.loc[:, "city"]))
relevant_train_labels = train_labels.loc[:, "total_cases"]
train_combined = pd.concat([train_values, relevant_train_labels], axis=1)

train_combined.loc[:, "type"] = "train"
test_values.loc[:, "type"] = "test"
data = pd.concat([train_combined, test_values], axis=0).reset_index(drop=True)

data_dict = {}
for city in cities:
    bool_city = data.loc[:, "city"] == city
    subset_data = data.loc[bool_city, :].reset_index(drop=True)
    data_dict[city] = subset_data

# %% Checking whether we need two different models

"""
Given that we have to predict cases for two different cities, it might
be valid to ask whether we need two different prediction models, or whether
one model would be sufficient for both.

Having one model would be helpful since it allows us to use more data for
training purposes. On the other hand, if the target variables of the
two cities are significantly different, we cannot train one model, given
that the underlying distribution has different properties.

In order to check what to do we start by applying a Kolmogrov-Smirnov test
on the raw target variable of both cities.
"""


def difference_in_distribution(series1, series2, file_text):

    # CDF
    def ecdf(data):
        """ Compute ECDF """
        x = np.sort(data)
        n = x.size
        y = np.arange(1, n+1) / n
        return x, y

    test_results = pd.DataFrame(index=["KS 2 Sample Test", "ANOVA"],
                                columns=["Statistic", "P Value"])
    test_results.iloc[0, :] = st.ks_2samp(series1, series2)
    test_results.iloc[1, :] = st.f_oneway(series1, series2)

    fig, axs = plt.subplots(ncols=3, figsize=(40, 15))

    # Time series
    axs[0].plot(np.arange(len(series1)), series1, color="b",
                label=series1.name)
    axs[0].plot(np.arange(len(series2)), series2, color="r",
                label=series2.name)
    axs[0].legend(prop={"size": 25})

    # Boxplots
    axs[1].boxplot([series1, series2])
    axs[1].set_xticklabels([series1.name, series2.name],
                           fontsize=25,
                           rotation=45)

    x, y = ecdf(series1)
    axs[2].scatter(x, y, color="b", label=series1.name)
    x, y = ecdf(series2)
    axs[2].scatter(x, y, color="r", label=series2.name)
    axs[2].legend(prop={"size": 25})

    for ax in axs.ravel():
        ax.tick_params(axis="both", labelsize=20)

    fig.tight_layout()
    fig.savefig(r"{}/{}.png".format(output_path, file_text),
                bbox_inches="tight")

    return test_results


sj_total_cases = data_dict["sj"].loc[:, "total_cases"].dropna()
sj_total_cases.name = "SJ"

iq_total_cases = data_dict["iq"].loc[:, "total_cases"].dropna()
iq_total_cases.name = "IQ"

difference_in_distribution(sj_total_cases, iq_total_cases,
                           "overall_distributions")

# %% Imputation class

"""
Text about data imputation
"""


class ImputingData():

    def __init__(self, data, features, model, threshold, city_name):
        self.data = data
        self.features = features
        self.threshold = threshold
        self.city_name = city_name
        self.model = model

        # Finding which columns are easy/ difficult to fill
        (self.easy_columns,
         self.diff_columns) = self.pct_missing_data(self.data,
                                                    self.features,
                                                    self.threshold,
                                                    self.city_name)

        # Assessing the strength of each imputation method
        self.nrmse_df = self.imputation_table(self.data,
                                              self.features,
                                              self.city_name)

        # Fill easy columns
        self.easy_filled_df = self.fill_easy_columns(self.data,
                                                     self.easy_columns,
                                                     self.nrmse_df)

        # Fill difficult columns
        (self.imputed_data,
         self.adj_nrmse_df) = self.fill_diff_columns(self.easy_filled_df,
                                                     self.model,
                                                     self.diff_columns,
                                                     self.easy_columns,
                                                     self.nrmse_df,
                                                     self.city_name)

    def pct_missing_data(self, data, cols, threshold, city_name):
        """
        This method does two things. First, it creates a chart showing the
        percentages of missing data. Second, it returns which columns
        have less than a certain threshold percentage of data, and which
        columns have more.

        Parameters
        ----------
        data : DataFrame
            DataFrame containing all the data
        cols : list
            List containing all the columns we have to fill
        threshold : int
            The threshold which divides the easy and difficult columns
        city_name : str
            A string containing the city name for which we calculate

        Returns
        -------
        easy_columns : list
            List containing the columns which have less then the threshold
            missing data
        diff_columns : list
            List containing the columns which have more than the threshold
            missing data

        """

        # Calculating the percentage missing
        num_of_obs = len(data)
        num_of_nans = data.loc[:, cols].isna().sum()
        df_pct_missing = pd.DataFrame(num_of_nans
                                      / num_of_obs*100).reset_index()
        df_pct_missing.rename(columns={"index": "columns",
                                       0: "pct_missing"}, inplace=True)
        df_sorted_values = df_pct_missing.sort_values(by="pct_missing",
                                                      ascending=False)

        # Column division
        bool_easy = df_sorted_values.loc[:, "pct_missing"] < threshold
        easy_columns = df_sorted_values.loc[bool_easy, "columns"]
        diff_columns = list(set(cols) - set(easy_columns))

        # Plotting the data
        fig, axs = plt.subplots(figsize=(20, 10))
        axs.bar(df_sorted_values.loc[:, "columns"],
                df_sorted_values.loc[:, "pct_missing"])
        axs.tick_params(axis="both", labelsize=16, labelrotation=90)
        axs.set_ylabel("Percentage of missing observations", fontsize=18)
        axs.set_xlabel("Variables", fontsize=18)
        axs.axhline(threshold, linestyle="dashed", color="red",
                    label="{} percent threshold".format(threshold))
        axs.legend(prop={"size": 20})
        fig.savefig(r"{}/{}_miss_data_pct.png".format(output_path, city_name),
                    bbox_inches="tight")

        return easy_columns, diff_columns

    def nrmse(self, y_true, y_pred, n):
        """
        This function calculates the normalized root mean squared error.

        Parameters
        ----------
        y_true : array
            The true values
        y_pred : array
            The predictions
        n : int
            The number of rows we testing for performance

        Returns
        -------
        rounded_nrmse : float
            The resulting, rounded nrmse

        """
        ts_min, ts_max = np.min(y_true), np.max(y_true)
        mse = sum((y_true-y_pred)**2) / n
        nrmse_value = np.sqrt(mse) / (ts_max-ts_min)
        rounded_nrmse = np.round(nrmse_value, 2)
        return rounded_nrmse

    def knn_mean(self, ts, n):
        """
        This function calculates the mean value of the n/2 values before
        and after it. This approach is therefore called the k nearest
        neighbour approach.

        Parameters
        ----------
        ts : array
            The time series we would like to impute
        n : int
            The number of time period before + after we would like
            to take the mean of

        Returns
        -------
        out : array
            The filled up time series.

        """
        out = np.copy(ts)
        for i, val in enumerate(ts):
            if np.isnan(val):
                n_by_2 = np.ceil(n/2)
                lower = np.max([0, int(i-n_by_2)])
                upper = np.min([len(ts)+1, int(i+n_by_2)])
                ts_near = np.concatenate([ts[lower:i], ts[i:upper]])
                out[i] = np.nanmean(ts_near)
        return out

    def imputation_table(self, data, cols, city_name):
        """
        This method calculates the nrmse for all columns and inserts them
        in a table. Additionally a graph is plotted in order for visual
        inspection afterwards. The score is calculated by randomly dropping
        50 values and then imputing them. Afterwards the performance is
        assessed.

        Parameters
        ----------
        data : DataFrame
            Dataframe which includes all columns
        cols : list
            List of columns we would like to impute
        city_name : str
            In order to know which city data was used, we specify the name

        Returns
        -------
        nrmse_df : DataFrame
            The results of each method for each column.

        """

        nrmse_df = pd.DataFrame(index=cols)
        print("Create imputation table")
        for col in tqdm(cols):

            original_series = data.loc[:, col]
            time_series = original_series.dropna().reset_index(drop=True)

            n = 50
            random.seed(42)
            rand_num = random.sample(range(0, len(time_series)), n)

            time_series_w_nan = time_series.copy()
            time_series_w_nan[rand_num] = np.nan

            # Forward fill ----
            ts_ffill = time_series_w_nan.ffill()
            nrmse_df.loc[col, "ffill"] = self.nrmse(time_series, ts_ffill, n)

            # Backward fill ----
            ts_bfill = time_series_w_nan.bfill()
            nrmse_df.loc[col, "bfill"] = self.nrmse(time_series, ts_bfill, n)

            # Linear Interpolation ----
            ts_linear = time_series_w_nan.interpolate(method="linear")
            nrmse_df.loc[col, "linear"] = self.nrmse(time_series,
                                                     ts_linear, n)

            # Cubic Interpolation ----
            ts_cubic = time_series_w_nan.interpolate(method="cubic")
            nrmse_df.loc[col, "cubic"] = self.nrmse(time_series, ts_cubic, n)

            # Mean of k nearest neighbours ----
            ts_knn = self.knn_mean(time_series_w_nan, 8)
            nrmse_df.loc[col, "knn"] = self.nrmse(time_series, ts_knn, n)

        # Plotting results
        adj_df = nrmse_df.reset_index()
        long_format = pd.melt(adj_df, id_vars=["index"], var_name=["nrmse"])
        fig, axs = plt.subplots(figsize=(20, 10))
        sns.barplot(x="index", y="value", hue="nrmse",
                    data=long_format, ax=axs)
        axs.tick_params(axis="both", labelsize=16, labelrotation=90)
        axs.set_ylabel("Normalized Root Mean Squared Root Error", fontsize=18)
        axs.set_xlabel("Variables", fontsize=18)
        axs.legend(prop={"size": 20})
        fig.savefig(r"{}/{}_imput_performance.png".format(output_path,
                                                          city_name),
                    bbox_inches="tight")

        return nrmse_df

    def fill_by_method(self, original_series, method):
        """
        After we know what the best method is for each column, we would
        like to impute the missing values. This function lists all
        potential methods, except the model build one.

        Parameters
        ----------
        original_series : array
            The original array with all its missing values
        method : str
            A string describing the best working method

        Returns
        -------
        time_series : array
            The original array now filled the missing values with the
            method of choice

        """

        if method == "ffill":
            time_series = original_series.ffill()
        elif method == "bfill":
            time_series = original_series.bfill()
        elif method == "linear":
            time_series = original_series.interpolate(method="linear")
        elif method == "cubic":
            time_series = original_series.interpolate(method="cubic")
        elif method == "knn":
            time_series = self.knn_mean(original_series, 8)
        return time_series

    def fill_easy_columns(self, data, easy_columns, nrmse_df):
        """
        This method goes through all easy declared columns and fills them
        up

        Parameters
        ----------
        data : Dataframe
            DataFrame containing all columns
        easy_columns : list
            List of all columns which can undergo the easy imputation
        nrmse_df : DataFrame
            Dataframe which contains the performance metrices of
            all imputation methods

        Returns
        -------
        data : Dataframe
            Dataframe with imputated columns

        """

        print("Filling easy columns")
        for col in tqdm(easy_columns):
            time_series = data.loc[:, col]
            best_method = nrmse_df.loc[col, :].sort_values().index[0]
            ts_filled = self.fill_by_method(time_series, best_method)
            data.loc[:, col] = ts_filled

            assert sum(data.loc[:, col].isna()) == 0, \
                "Easy imputation went wrong"
        return data

    def fill_diff_columns(self, data, model, diff_columns,
                          easy_columns, nrmse_df, city_name):
        """
        This method imputes the difficult columns. Difficult means that
        these columns miss more than the specified threshold percentage
        of observations. Because of that a model based approach is tried.
        If this approach proves better than the normal methods, it is
        applied.
        Furthermore, we plot the nrmse of the model based approach in order
        to compare these with the normal methods

        Parameters
        ----------
        data : DataFrame
            Dataframe containing all data
        model : dictionary
            Here we specify the model and the parameters we would like to try
        diff_columns : list
            List of columns we would like to try
        easy_columns : list
            List of columns which have less than the threshold percentage
            data missing
        nrmse_df : Dataframe
            Dataframe with the nrmse for all methods and columns
        city_name : str
            String specifying which city we are talking about

        Returns
        -------
        data : Dataframe
            Dataframe with imputated columns
        diff_nrmse_df : Dataframe
            Dataframe showing the nrmse performance of the difficult
            columns and all methods

        """
        non_knn_method = list(set(nrmse_df.columns) - set(["knn"]))
        diff_nrmse_df = nrmse_df.loc[diff_columns, non_knn_method]
        print("Filling difficult columns")
        for col in tqdm(diff_columns):

            # Getting data ready
            time_series = data.loc[:, col]
            non_nan_data = data.dropna(subset=[col])
            features = non_nan_data.loc[:, easy_columns]
            scaler = StandardScaler().fit(features)
            scaled_features = scaler.transform(features)
            target = non_nan_data.loc[:, col]

            # Model building and evaluation
            model_file_name = "{}/{}_{}_model.pickle".format(data_path,
                                                             city_name,
                                                             col)
            if not os.path.isfile(model_file_name):
                model_info = model_train(model, scaled_features,
                                         target,
                                         "neg_mean_squared_error",
                                         False)
                with open(model_file_name, "wb") as file:
                    pickle.dump(model_info, file)
            else:
                with open(model_file_name, "rb") as f:
                    model_info = pickle.load(f)
            target_min, target_max = np.min(target), np.max(target)
            rmse = np.sqrt(abs(model_info["scores"][0]))
            nrmse_value = rmse / (target_max-target_min)
            diff_nrmse_df.loc[col, "model"] = nrmse_value

            # Imputing the difficult ones
            argmin_method = np.argmin(diff_nrmse_df.loc[col, :])
            best_method = diff_nrmse_df.columns[argmin_method]
            bool_target_nan = time_series.isna()
            if best_method == "model":
                features = data.loc[bool_target_nan, easy_columns]
                scaled_features = scaler.transform(features)
                pred = model_info["model"].predict(scaled_features)
                data.loc[bool_target_nan, col] = pred
            else:
                pred = self.fill_by_method(time_series, best_method)
                data.loc[bool_target_nan, col] = pred

        assert data.loc[:, list(easy_columns) + diff_columns]\
            .isna().any().any() == False, "Still missing data"

        # Plotting results
        adj_df = diff_nrmse_df.reset_index()
        long_format = pd.melt(adj_df, id_vars=["index"], var_name=["nrmse"])
        fig, axs = plt.subplots(figsize=(20, 10))
        sns.barplot(x="index", y="value", hue="nrmse",
                    data=long_format, ax=axs)
        axs.tick_params(axis="both", labelsize=16, labelrotation=90)
        axs.set_ylabel("Normalized Root Mean Squared Root Error", fontsize=18)
        axs.set_xlabel("Variables", fontsize=18)
        axs.legend(prop={"size": 20})
        fig.savefig(r"{}/{}_diff_columns.png".format(output_path, city_name),
                    bbox_inches="tight")

        return data, diff_nrmse_df


# %% Seasonal and Trend Class


class TrendSeasonality():

    def __init__(self, data, features, city_name):
        self.data = data
        self.features = features
        self.city_name = city_name
        self.features_w_target = features + ["total_cases"]

        # Finding the potential seasonality spikes
        self.spike_dict = self.spike_finder(self.data,
                                            self.features,
                                            self.city_name)

        # Looking at the autocorrelation plot
        self.acf_plots(self.data, self.features,
                       self.spike_dict, self.city_name)

        # Now we remove the seasonality and potential trend for these columns
        self.removed_season_data = self.season_trend(self.data,
                                                     self.features,
                                                     self.spike_dict)

        # Now we check which of the other variables could have a potential
        # trend. If they have one, we remove it
        (self.detrended_data,
         self.trend_dict) = self.trend_detecter(self.removed_season_data,
                                                self.features)

        # a Dickey Fuller test - which we expect to be significant for all
        self.adf_dict = self.dickey_fuller_test(self.detrended_data,
                                                self.features_w_target)

        # Those which exhibit a unit root, have to be differenced
        (self.final_data,
         self.adf_dict_post,
         self.diff_dict) = self.diff_sign_columns(self.detrended_data,
                                                  self.features,
                                                  self.adf_dict)

        # We now look at the distribution of the variables
        self.plotting_line_dist(self.final_data, self.features,
                                self.city_name)

    def spike_finder(self, data, cols, city_name):
        """
        This method calculates the power-plots for all specified
        variables. Afterwards spikes above a certain threshold and
        which exhibit the desired prominence are marked. Afterwards
        an image of all columns is saved

        Parameters
        ----------
        data : DataFrame
            Dataframe containing all the columns for which we would
            like to calculate the power-plots of
        cols : list
            Columns which we would like to examine
        city_name : str
            A string denoting which city we are looking at

        Returns
        -------
        spikes_dict : dict
            Dictionary which saves the dominant and prominent
            frequencies for each exogenous variables

        """
        fig, axs = plt.subplots(nrows=4, ncols=5, figsize=(20, 20),
                                sharex=True)
        plt.subplots_adjust(right=None)
        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor="none", top=False, bottom=False,
                        left=False, right=False)
        plt.grid(False)
        plt.xlabel("Frequency [1 / Hour]", fontsize=22, labelpad=20)
        plt.ylabel("Amplitude", fontsize=22, labelpad=50)
        spikes_dict = {}
        axs = axs.ravel()
        for i, col in enumerate(cols):

            signal = data.loc[:, col].copy()
            fft_output = fft.fft(signal.values)
            power = np.abs(fft_output)
            freq = fft.fftfreq(len(signal))

            mask = freq > 0
            pos_freq = freq[mask]
            power = power[mask]

            axs[i].plot(pos_freq, power)
            axs[i].tick_params(axis="both", labelsize=16)
            axs[i].set_title(col, fontsize=12)
            
            relevant_power = power[:int(len(power)/4)]
            prominence = np.mean(relevant_power) * 5
            threshold = np.mean(relevant_power) + 3 * np.std(relevant_power)
            peaks = sig.find_peaks(relevant_power, prominence=prominence,
                                   threshold=threshold)[0]
            peak_freq = pos_freq[peaks]
            peak_power = power[peaks]
            axs[i].plot(peak_freq, peak_power, "ro")
            if len(peak_freq) > 0:
                spikes_dict[col] = (1/peak_freq).tolist()[0]
            else:
                spikes_dict[col] = np.nan

        fig.tight_layout()
        fig.savefig(r"{}/{}_fourier_transform.png".format(output_path,
                                                          city_name),
                    bbox_inches="tight")
        return spikes_dict

    def acf_plots(self, data, cols, spike_dict, city_name):
        """
        This method plots the autocorrelation functions for all
        specified columns in a specified dataframe. Furthermore,
        the biggest possible spike for each column, if there is any,
        is made visible through a vertical line and a legend

        Parameters
        ----------
        data : DataFrame
            The dataframe which contains all exogenous variables.
        cols : list
            A list containing the columns which should be
            analysed
        spike_dict : dict
            A dictionary having all columns as the keys and the
            potential spike as the value
        city_name : str
            A string to save the resulting png properly

        Returns
        -------
        None.

        """
        fig, axs = plt.subplots(nrows=4, ncols=5, sharex=True, sharey=True,
                                figsize=(20, 20))
        plt.subplots_adjust(right=None)
        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor="none", top=False, bottom=False,
                        left=False, right=False)
        plt.grid(False)
        plt.xlabel("Lags", fontsize=22, labelpad=20)
        plt.ylabel("Correlation", fontsize=22, labelpad=50)
        axs = axs.ravel()
        max_lags = round(np.nanmax(list(spike_dict.values())))
        for i, col in enumerate(cols):
            series = data.loc[:, col].copy()
            sm.graphics.tsa.plot_acf(series.values.squeeze(),
                                     lags=max_lags, ax=axs[i], missing="drop")
            axs[i].set_title(col, fontsize=12)
            axs[i].tick_params(axis="both", labelsize=16)
            if not np.isnan(spike_dict[col]):
                axs[i].axvline(spike_dict[col], -1, 1, color="red",
                               label="Periodicity: {}".format(spike_dict[col]))
                axs[i].legend(loc="upper center", prop={'size': 16})
        fig.tight_layout()
        fig.savefig(r"{}/{}_autocorrelation_function.png".format(output_path,
                                                                 city_name),
                    bbox_inches="tight")

    def season_trend(self, data, cols, spike_dict):
        """
        This method decomposes the time series by removing
        (subtracting) the modelled seasonality and trend.

        Parameters
        ----------
        data : DataFrame
            A dataframe containing the relevant time series
        cols : list
            A list which specifies all potentially affected columns
        spike_dict : dict
            A dictionary stating the significant seasonality for
            each column

        Returns
        -------
        data : Dataframe
            After decomposing and 'cleaning', we put the variables
            back into the dataframe which is returned

        """
        for col in cols:
            period = spike_dict[col]
            time_series = data.loc[:, col]
            if not np.isnan(period):
                res = STL(time_series, period=int(spike_dict[col])+1).fit()
                adjusted_series = time_series - res.seasonal
                data.loc[:, col] = adjusted_series

        return data

    def trend_detecter(self, data, cols):
        """
        This method tests for a deterministic trend using the
        Mann-Kendall test. If the test is found to be significant,
        the trend is removed (subtracted).

        Parameters
        ----------
        data : DataFrame
            A dataframe containing all the relevant columns
        cols : list
            A list of column names for which we apply the test

        Returns
        -------
        no_nan_data : DataFrame
            A dataframe with the potentially removed trend series
        trend_dict : dict
            A dictionary containing the information of the detrending

        """
        trend_dict = {}
        for col in cols:
            trend_dict[col] = {}

            time_series = data.loc[:, col]
            result = mk.original_test(time_series)
            trend_dict[col]["pre_trend"] = result.trend

            if result.trend != "no trend":
                d_trend = [(result.intercept + result.slope * x)
                           for x in np.arange(len(time_series))]
                trend_dict[col]["intercept"] = result.intercept
                trend_dict[col]["slope"] = result.slope

                adj_time_series = time_series - d_trend
                result = mk.original_test(adj_time_series)
                trend_dict[col]["post_trend"] = result.trend
                data.loc[:, col] = adj_time_series

        no_nan_data = data.dropna(subset=cols).reset_index(drop=True)
        return no_nan_data, trend_dict

    def dickey_fuller_test(self, data, cols):
        """
        Method to test certain rows from a dataframe whether
        an unit root is present through the ADF test

        Parameters
        ----------
        data : Dataframe
            A dataframe which contains all series we would like to
            test
        cols : list
            A list containing all columns names for which we would
            like to conduct the test for.

        Returns
        -------
        adf_dict : dict
            Dictionary containing the test result for every series.

        """
        adf_dict = {}
        for col in cols:
            time_series = data.loc[:, col].dropna()
            result = adfuller(time_series, autolag="AIC", regression="c")
            adf_dict[col] = result[1]
        return adf_dict

    def diff_sign_columns(self, data, cols, adf_dict):
        """
        This method differences the time series if a non significant
        dickey fuller test is shown. This is done as long as the
        adf is not significant.

        Parameters
        ----------
        data : Dataframe
            A dataframe containing all the time series we would like to test
        cols : list
            List of column names we would like to test
        adf_dict : dict
            dictionary containing the test results of the dickey fuller test

        Returns
        -------
        data : DataFrame
            A dataframe with the now potentially differenced series
        adf_dict : dict
            A dictionary with the now significant dickey fuller tests
        number_of_diff : dict
            A dictionary telling how often each series was differenced.

        """
        number_of_diff = {}
        for col in cols:
            pvalue = adf_dict[col]
            time_series = data.loc[:, col].dropna()
            while pvalue > 0.05:
                time_series = time_series.diff(periods=1)
                pvalue = adfuller(time_series.dropna(),
                                  autolag="AIC",
                                  regression="c")[1]
                number_of_diff[col] = sum(time_series.isna())
            adf_dict[col] = pvalue
            data.loc[:, col] = time_series

        return data, adf_dict, number_of_diff

    def plotting_line_dist(self, data, cols, city_name):
        nrows, ncols = 4, 5
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=True,
                                figsize=(ncols*10, nrows*10))
        axs = axs.ravel()
        for i, col in enumerate(cols):
            axs[i].plot(data.loc[:, col], color="b")
            axs[i].set_title(col, fontsize=40)
            axs[i].tick_params(axis="both", labelsize=30)
            axs[i].set_xlabel("")
        fig.tight_layout()
        fig.savefig(r"{}/{}_lineplot.png".format(output_path,
                                                 city_name),
                    bbox_inches="tight")

        nrows, ncols = 4, 5
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols,
                                figsize=(ncols*10, nrows*10))
        axs = axs.ravel()
        for i, col in enumerate(cols):
            sns.distplot(data.loc[:, col],
                         color="b", ax=axs[i])
            axs[i].set_title(col, fontsize=40)
            axs[i].tick_params(axis="both", labelsize=30)
            axs[i].set_xlabel("")
        fig.tight_layout()
        fig.savefig(r"{}/{}_distplot.png".format(output_path,
                                                 city_name),
                    bbox_inches="tight")


# %% Processing the data

transformed_data_dict = {}
for city in cities:
    # Get example data
    data = data_dict[city].copy()

    # Imputing the data
    imputer = ImputingData(data.copy(), features, imputation_model, 1, city)
    imputed_data = imputer.imputed_data.copy()

    # Transforming the data
    transformer = TrendSeasonality(imputed_data, features, city)
    transformed_data = transformer.final_data.copy()
    transformed_data_dict[city] = transformed_data

# %% Functions for estimation


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


def plot_cutoff_comparison(time_series, cutoff, file_text):

    first_half = time_series[:cutoff]
    first_half.name = "First Half"
    second_half = time_series[cutoff:]
    second_half.name = "Second Half"
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))
    axs[0].plot(time_series, label="Complete Series")
    axs[0].axvline(cutoff, color="r", linestyle="--", label="Cutoff")
    axs[1].plot(second_half,
                label="Series from observation {} onwards".format(cutoff))
    axs = axs.ravel()
    for axes in axs.ravel():
        axes.legend(prop={"size": 16}, loc="upper right")
        axes.tick_params(axis="both", labelsize=16)
    fig.tight_layout()
    fig.savefig(r"{}/{}.png".format(output_path, file_text),
                bbox_inches="tight")
    return first_half, second_half


def highest_seasonality(series, city_name):
    fig, axs = plt.subplots(figsize=(10, 10))
    signal = series.dropna()
    fft_output = fft.fft(signal.values)
    power = np.abs(fft_output)
    freq = fft.fftfreq(len(signal))

    mask = freq > 0
    pos_freq = freq[mask]
    power = power[mask]

    axs.plot(pos_freq, power)

    highest_power_level_pos = np.argmax(power)
    highest_power_freq = pos_freq[highest_power_level_pos]
    translated_frequency = int(round(1/highest_power_freq))
    axs.scatter(highest_power_freq,
                np.max(power),
                color="red",
                s=100,
                label="Seasonality: {}".format(round(translated_frequency, 4)))
    axs.legend(prop={"size": 16})
    fig.tight_layout()
    fig.savefig(r"{}/{}_total_cases_seasonality.png".format(output_path,
                                                            city_name),
                bbox_inches="tight")

    return translated_frequency


def lasso_relevant_columns_finder(data):

    not_needed_columns = ["city", "type", "index",
                          "week_start_date", "year", "weekofyear"]
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
                                      "scoring": "neg_mean_absolute_error"})
    bool_non_null = lasso_model_info["model"].coef_ != 0
    lasso_chosen_columns = list(X_data.columns[bool_non_null])
    return lasso_chosen_columns


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
    fig.savefig(r"{}/{}_target_acf_pacf.png".format(output_path, city_name),
                bbox_inches="tight")


# %% Prediction for SJ

### Investigation

data = transformed_data_dict["sj"].copy()
bool_target_na = data.loc[:, "total_cases"].isna()

"""
Given that the time series exhibits a structural break around half of the
dataset, we for that
"""

cutoff = int(sum(~bool_target_na)/2)
y_time_series = data.loc[~bool_target_na, "total_cases"]
first_half, second_half = plot_cutoff_comparison(y_time_series,
                                                 cutoff, "sj_cutoff")
difference_in_distribution(first_half, second_half, "test_sj")
cut_data = data.loc[cutoff:, :]
bool_target_na = cut_data.loc[:, "total_cases"].isna()

"""

"""

train_cutoff = 0.8
train_nrows = int(max(cut_data.dropna().index) * train_cutoff)


### TODO Plot what we would like to do here with vertical lines

### SARIMAX Cut version

"""
Given the unsignificant pvalue, no differencing is needed. Now we have
to estimate the number of of p and q within the ARMA model. We start
by plotting the ACF and PACF model to see which range values we would like
to test for.
"""

n_years = 4
number_of_weeks_per_year = 52
nlags = int(n_years * number_of_weeks_per_year)
cut_y_time_series = cut_data.loc[~bool_target_na, "total_cases"]
acf_pacf_plots(cut_y_time_series, nlags, "sj_cut")

jitter_factor = 1
fig, axs = plt.subplots(figsize=(10, 10))
sm.graphics.tsa.plot_acf(cut_y_time_series, lags=nlags, fft=True, ax=axs)
for i in np.arange(1, n_years+1):
    axs.axvline(i * number_of_weeks_per_year - jitter_factor,
                color="r", linestyle="--", linewidth=4)
for i in np.arange(1, n_years/2+1):
    axs.axvline(i * (2*number_of_weeks_per_year) + jitter_factor,
                color="g", linestyle="--", linewidth=4)
axs.tick_params(axis="both", labelsize=16)
axs.set_title("ACF with 52 and 104 weeks vertical lines",
              fontsize=16)
fig.tight_layout()
fig.savefig(r"{}/sj_cut_acf_investigation.png".format(output_path),
            bbox_inches="tight")

"""
Now we would like to see whether there would be any gain to include lags
of other independent variables. We do that by using a granger causality test
"""

cut_lasso_columns = lasso_relevant_columns_finder(cut_data.copy())

"""
Training the plain vanilla SARIMAX with yearly seasonality. Furthermore,
we adjust some hyperparameters, namely p and q. Given the PACF is zero
after the second lag, we do not expect the AR term to be much higher
than three. The MA term is a bit more tricky given that the ACF decays
exponentially, though the MA term does not really oscillate much. We
therefore try a couple of more parameters for the MA term, namely 8.
"""

cut_train_endog = cut_data.loc[:train_nrows, "total_cases"]
model_file_name = "{}/cut_sarimax_sj.pickle".format(data_path)
if not os.path.isfile(model_file_name):

    max_ar = list(np.arange(0, 3+1))
    max_ma = list(np.arange(0, 10+1))
    seasons = [52, 104]

    best_aic_so_far = float("inf")
    for season in tqdm(seasons):
        cut_aic_df = pd.DataFrame(np.zeros((len(max_ar), len(max_ma))))
        for p in tqdm(max_ar):
            for q in tqdm(max_ma):
                # Fit the model
                model = sm.tsa.statespace.SARIMAX(
                    endog=cut_train_endog,
                    order=(p, 0, q),
                    seasonal_order=(1, 0, 1, season),
                    trend="c")
                try:
                    res = model.fit(disp=False)
                    cut_aic_df.iloc[p, q] = res.aic
                except:
                    cut_aic_df.iloc[p, q] = np.nan

                # Save best model until now
                if res.aic < best_aic_so_far:
                    best_aic_so_far = res.aic
                    best_cut_sarimax_model = res

        # Create heatmaps out of all dataframes
        fig, axs = plt.subplots(figsize=(20, 10))
        sns.heatmap(cut_aic_df, annot=True, fmt=".1f",
                    ax=axs, annot_kws={"size": 14},
                    vmin=np.min(cut_aic_df.values),
                    vmax=np.percentile(cut_aic_df.values, 25))
        axs.set_xlabel("MA Terms", fontsize=18)
        axs.set_ylabel("AR Terms", fontsize=18)
        axs.tick_params(axis="both", labelsize=16)
        fig.tight_layout()
        fig.savefig(r"{}/sj_cut_sarimax_{}.png".format(output_path, season),
                    bbox_inches="tight")

    with open(model_file_name, "wb") as file:
        pickle.dump(best_cut_sarimax_model, file)
else:
    with open(model_file_name, "rb") as f:
        best_cut_sarimax_model = pickle.load(f)

"""
Now we analyze the error term of the best model.
"""

fig = best_cut_sarimax_model.plot_diagnostics(figsize=(20, 20))
fig.tight_layout()
fig.savefig(r"{}/sj_cut_sarimax_errors.png".format(output_path),
            bbox_inches="tight")

### SARIMAX Full version

"""
Finding correct seasonality to specify. We do that by appling a fourier
transform and look at the most dominating wavelength if we find one at all.
"""

seasonal_y = highest_seasonality(y_time_series, "sj")
acf_pacf_plots(y_time_series, nlags, "sj_full")

"""
Now we would like to see whether there would be any gain to include lags
of other independent variables. We do that by using a granger causality test
"""

full_lasso_columns = lasso_relevant_columns_finder(data.copy())

"""

"""


train_data = data.loc[:train_nrows, :]
test_data = data.loc[train_nrows:, :].dropna()

full_exog = data.loc[:, full_lasso_columns]
full_scaler = StandardScaler().fit(full_exog)

strain_exog_full = full_scaler.transform(
    train_data.loc[:, full_lasso_columns])
stest_exog_full = full_scaler.transform(
    test_data.loc[:, full_lasso_columns])

full_train_endog = data.loc[:train_nrows, "total_cases"]

model_file_name = "{}/full_sarimax_sj.pickle".format(data_path)
if not os.path.isfile(model_file_name):
    max_ar = list(np.arange(0, 5+1))
    max_ma = list(np.arange(0, 10+1))
    full_aic_df = pd.DataFrame(np.zeros((len(max_ar), len(max_ma))))
    best_aic_so_far = float("inf")
    for p in tqdm(max_ar):
        for q in tqdm(max_ma):
            # Fit the model
            model = sm.tsa.statespace.SARIMAX(
                endog=full_train_endog, exog=strain_exog_full,
                order=(p, 0, q),
                time_varying_regression=True,
                mle_regression=False,
                seasonal_order=(1, 0, 1, seasonal_y),
                trend="c")
            try:
                res = model.fit(disp=False)
                full_aic_df.iloc[p, q] = res.aic
            except:
                full_aic_df.iloc[p, q] = np.nan

            # Save best model until now
            if res.aic < best_aic_so_far:
                best_aic_so_far = res.aic
                best_full_sarimax_model = res

    # Create heatmaps out of all dataframes
    fig, axs = plt.subplots(figsize=(20, 10))
    sns.heatmap(full_aic_df, annot=True, fmt=".1f",
                ax=axs, annot_kws={"size": 14},
                vmin=np.min(full_aic_df.values),
                vmax=np.percentile(full_aic_df.values, 25))
    axs.set_xlabel("MA Terms", fontsize=18)
    axs.set_ylabel("AR Terms", fontsize=18)
    axs.tick_params(axis="both", labelsize=16)
    fig.tight_layout()
    fig.savefig(r"{}/sj_full_sarimax_{}.png".format(output_path, seasonal_y),
                bbox_inches="tight")

    with open(model_file_name, "wb") as file:
        pickle.dump(best_full_sarimax_model, file)
else:
    with open(model_file_name, "rb") as f:
        best_full_sarimax_model = pickle.load(f)

fig = best_full_sarimax_model.plot_diagnostics(figsize=(20, 20))
fig.tight_layout()
fig.savefig(r"{}/sj_full_sarimax_errors.png".format(output_path),
            bbox_inches="tight")


###  TBATS

"""
Finally we try Tbats also for all seasonalities and for both time
lengths. As we did for the SARIMAX models, we use the AIC criteria
in order to find the superior model.
"""

tbats_seasonalities = {"No Seasonality": None,
                       "Yearly": [number_of_weeks_per_year],
                       "Every other year": [2*number_of_weeks_per_year],
                       "Both": (number_of_weeks_per_year,
                                2*number_of_weeks_per_year)}
datalengths = {"Half": cut_train_endog,
               "Full": full_train_endog}

tbats_model_dict = {}
for data_key, data_value in tqdm(datalengths.items()):
    tbats_aic_df = pd.DataFrame(index=tbats_seasonalities.keys(),
                                columns=[data_key])
    lowest_aic = float("inf")
    for season_key, season_value in tbats_seasonalities.items():
        estimator = TBATS(seasonal_periods=season_value)
        tbats_model = estimator.fit(data_value)
        tbats_aic = tbats_model.aic
        tbats_aic_df.loc[season_key, data_key] = tbats_aic
        if tbats_model.aic < lowest_aic:
            lowest_aic = tbats_model.aic
            tbats_model_dict[data_key] = tbats_model
    # Create heatmaps out of all dataframes
    fig, axs = plt.subplots(figsize=(10, 10))
    sns.heatmap(tbats_aic_df.astype(float), annot=True, fmt=".1f",
                ax=axs, annot_kws={"size": 18},
                vmin=np.min(tbats_aic_df.values),
                vmax=np.percentile(tbats_aic_df.values, 25))
    axs.set_xlabel("Amount of time series used", fontsize=18)
    axs.set_ylabel("Seasonalities adjusted for", fontsize=18)
    axs.tick_params(axis="both", labelsize=16, rotation=45)
    fig.tight_layout()
    fig.savefig(r"{}/sj_tbats_{}.png".format(output_path, data_key),
                bbox_inches="tight")

    tbats_residuals = tbats_model_dict[data_key].resid
    tbats_std_residuals = tbats_residuals / np.std(tbats_residuals)

    fig, ax = plt.subplots(ncols=2, figsize=(20, 10))
    sm.qqplot(tbats_std_residuals, line="45", fit=True, ax=ax[0])
    sm.graphics.tsa.plot_acf(tbats_std_residuals, lags=52, fft=True, ax=ax[1])
    for axs, title in zip(ax.ravel(),
                          ["QQ Plot", "Autocorrelation"]):
        axs.tick_params(axis="both", labelsize=16)
        axs.set_title(title, fontsize=16)
    fig.tight_layout()
    fig.savefig(r"{}/sj_tbats_model_errors_{}.png".format(output_path,
                                                          data_key),
                bbox_inches="tight")

### Model Comparison

"""
After we trained all models, it is now time to see which one performs best.
This is done by predicting the OOS observations. We use the MAE in order
to compare performance between models. Given that the FFT Tricked SARIMAX
favoured to not use any fourier terms, we will only use the simple
SARIMAX for comparison reasons.
"""

y_true = data.loc[train_nrows:, "total_cases"].dropna()

cut_sarimax_pred = best_cut_sarimax_model.forecast(len(y_true))
cut_sarimax_pred[cut_sarimax_pred < 0] = 0

full_sarimax_pred = best_full_sarimax_model.forecast(len(y_true),
                                                     exog=stest_exog_full)
full_sarimax_pred[full_sarimax_pred < 0] = 0

half_tbat_pred = tbats_model_dict["Half"].forecast(steps=len(y_true))
half_tbat_pred[half_tbat_pred < 0] = 0

full_tbat_pred = tbats_model_dict["Full"].forecast(steps=len(y_true))
full_tbat_pred[full_tbat_pred < 0] = 0


half_tbats_mae = mean_absolute_error(y_true, half_tbat_pred)
full_tbats_mae = mean_absolute_error(y_true, full_tbat_pred)
cut_sarimax_mae = mean_absolute_error(y_true, cut_sarimax_pred)
full_sarimax_mae = mean_absolute_error(y_true, full_sarimax_pred)

fig, ax = plt.subplots(figsize=(20, 10))
ax.plot(y_true.values, label="Actual")
ax.plot(half_tbat_pred,
        label="Half TBATS - MAE: {}".format(round(half_tbats_mae, 2)))
ax.plot(full_tbat_pred,
        label="Full TBATS - MAE: {}".format(round(full_tbats_mae, 2)))
ax.plot(cut_sarimax_pred.values,
        label="Cut SARIMAX - MAE: {}".format(round(cut_sarimax_mae, 2)))
ax.plot(full_sarimax_pred.values,
        label="Full SARIMAX - MAE: {}".format(round(full_sarimax_mae, 2)))
ax.legend(loc="upper center", prop={"size": 16})
ax.tick_params(axis="both", labelsize=16)
fig.tight_layout()
fig.savefig(r"{}/sj_model_comp.png".format(output_path),
            bbox_inches="tight")

### Best model statistics

"""
After finding out that XXX is the best model, we can now train the model
on the entirety of the data. We now try several more hyperparameters,
including also trying the two year seasonality. Before that we scale
the exogenous variables. Given that we already know how the variables
look like in the future, we fit them on the entirety and transform then
the relevant part.
"""

n_preds = sum(bool_target_na)
in_sample_len = len(y_time_series)

estimator = TBATS(seasonal_periods=[2*number_of_weeks_per_year])
best_tbats_model = estimator.fit(cut_y_time_series)
all_in_sample_pred = best_tbats_model.y_hat
oos_pred, confid_info = best_tbats_model.forecast(n_preds,
                                                  confidence_level=0.95)
oos_pred[oos_pred < 0] = 0

fig, ax = plt.subplots(figsize=(20,10))
cut_y_time_series.plot(ax=ax, style="o", label="Actual")
ax.plot(cut_y_time_series.index,
        all_in_sample_pred, color="r", linestyle="--",
        label="In Sample Predictions")
ax.plot(np.arange(in_sample_len, in_sample_len + n_preds),
        oos_pred, color="g", linestyle="-",
        label="Out of Sample Predictions")
ax.fill_between(np.arange(in_sample_len, in_sample_len + n_preds),
                confid_info["lower_bound"],
                confid_info["upper_bound"],
                color="g", alpha=0.2, label="95% Confidence Intervals")
ax.legend(loc="upper right", prop={"size": 20})
ax.tick_params(axis="both", labelsize=16)
fig.tight_layout()
fig.savefig(r"{}/sj_tbats_performance_final.png".format(output_path),
            bbox_inches="tight")

sj_pred = pd.DataFrame({
    "total_cases": np.round(oos_pred),
    "city": "sj",
    "year": data.loc[bool_target_na, "year"],
    "weekofyear": data.loc[bool_target_na, "weekofyear"],
    })

# %% Prediction for IQ

"""
Here we start
"""

### Investigation

data = transformed_data_dict["iq"].copy()
bool_target_na = data.loc[:, "total_cases"].isna()

"""
Looking at the graph we find two things. First the beginning of the time
series is full of zeros, which is not representative for the rest of the
series. Therefore we will cut that part off. Secondly, we find one spike
which is drastically higher than the other spikes. This spike could
potentially harm the model's performance. Therefore, we winsorize the data
to the 95% level
"""

y_time_series = data.loc[~bool_target_na, "total_cases"]
win_time_series = winsorizer(y_time_series, 0.5, "iq")

iq_cutpoint = 75
plot_cutoff_comparison(win_time_series, iq_cutpoint, "iq_cutting_off_zeroes")
data = data.loc[iq_cutpoint:, :].reset_index(drop=True)
bool_target_na = data.loc[:, "total_cases"].isna()
cut_y_time_series = data.loc[~bool_target_na, "total_cases"]

"""
Now we take a look at the ACF and PACF in order to see whether any
seasonality is detected
"""

nlags = number_of_weeks_per_year * 3
acf_pacf_plots(cut_y_time_series, nlags, "iq")

"""
There might be a yearly seasonality, though it is far from being statistically
significant. Given that we might check whether to use no or yearly
seasonality.

Now we check what could be potential exogenous variables to use. We do that
by appling a Lasso regression
"""

lasso_columns = lasso_relevant_columns_finder(data.copy())

"""
We find no variable to have any explanatory power in the lasso regression.
Given that we will apply an SARIMA model. Additionally we compare the
performance to a TBATS model which we also applied earlier. In order
to compare both models we have to split into train and test. As before,
we will use 80% of the data as trainings data.
"""

train_length = int(len(y_time_series) * 0.8)
train_data = y_time_series[:train_length]
y_true = y_time_series[train_length:]

### SARIMA

max_ar = list(np.arange(0, 5+1))
max_ma = list(np.arange(0, 8+1))
seasons = [(1, 0, 1, 52), (0, 0, 0, 0)]

model_file_name = "{}/sarimax_iq.pickle".format(data_path)
if not os.path.isfile(model_file_name):

    best_aic_so_far = float("inf")
    for season in tqdm(seasons):
        simple_aic_full = pd.DataFrame(np.zeros((len(max_ar), len(max_ma))))
        for p in tqdm(max_ar):
            for q in tqdm(max_ma):
                # Fit the model
                model = sm.tsa.statespace.SARIMAX(
                    endog=train_data,
                    order=(p, 0, q),
                    seasonal_order=season,
                    trend="c")
                try:
                    res = model.fit(disp=False)
                    simple_aic_full.iloc[p, q] = res.aic
                except:
                    simple_aic_full.iloc[p, q] = np.nan

                # Save best model until now
                if res.aic < best_aic_so_far:
                    best_aic_so_far = res.aic
                    best_basic_model = res

        # Create heatmaps out of all dataframes
        fig, axs = plt.subplots(figsize=(10, 10))
        sns.heatmap(simple_aic_full, annot=True, fmt=".1f",
                    ax=axs, annot_kws={"size": 14},
                    vmin=np.min(simple_aic_full.values),
                    vmax=np.percentile(simple_aic_full.values, 25))
        axs.set_xlabel("MA Terms", fontsize=18)
        axs.set_ylabel("AR Terms", fontsize=18)
        axs.tick_params(axis="both", labelsize=16)
        fig.tight_layout()
        fig.savefig(r"{}/iq_sarimax_{}.png".format(output_path,
                                                   season[3]),
                    bbox_inches="tight")

    with open(model_file_name, "wb") as file:
        pickle.dump(best_basic_model, file)
else:
    with open(model_file_name, "rb") as f:
        best_basic_model = pickle.load(f)

### TBATS

seasonal_per = [None, [number_of_weeks_per_year]]
lowest_aic = float("inf")
tbats_aic_dict = {}
for period in seasonal_per:
    estimator = TBATS(seasonal_periods=period)
    tbats_model = estimator.fit(train_data)
    tbats_aic = tbats_model.aic

    if period is None:
        tbats_aic_dict[period] = tbats_aic
    else:
        tbats_aic_dict[period[0]] = tbats_aic

    if tbats_model.aic < lowest_aic:
        lowest_aic = tbats_model.aic
        best_tbats_model = tbats_model

print(best_tbats_model.summary())
tbats_residuals = best_tbats_model.resid
tbats_std_residuals = tbats_residuals / np.std(tbats_residuals)

fig, ax = plt.subplots(ncols=2, figsize=(20, 10))
sm.qqplot(tbats_std_residuals, line="45", fit=True, ax=ax[0])
sm.graphics.tsa.plot_acf(tbats_std_residuals, lags=52, fft=True, ax=ax[1])
for axs, title in zip(ax.ravel(),
                      ["QQ Plot", "Autocorrelation"]):
    axs.tick_params(axis="both", labelsize=16)
    axs.set_title(title, fontsize=16)
fig.tight_layout()
fig.savefig(r"{}/iq_tbats_model_errors.png".format(output_path),
            bbox_inches="tight")

### Model Comparison

simple_sarimax_pred = best_basic_model.forecast(len(y_true))
simple_sarimax_pred[simple_sarimax_pred < 0] = 0
tbat_pred = best_tbats_model.forecast(steps=len(y_true))

tbats_mae = mean_absolute_error(y_true, tbat_pred)
ssarimax_mae = mean_absolute_error(y_true, simple_sarimax_pred)

fig, ax = plt.subplots(figsize=(20, 10))
ax.plot(y_true.values, label="Actual")
ax.plot(tbat_pred, label="TBATS - MAE: {}".format(round(tbats_mae, 2)))
ax.plot(simple_sarimax_pred.values,
        label="Simple SARIMA - MAE: {}".format(round(ssarimax_mae, 2)))
ax.legend(loc="upper center", prop={"size": 16})
ax.tick_params(axis="both", labelsize=16)
fig.tight_layout()
fig.savefig(r"{}/iq_model_comp.png".format(output_path),
            bbox_inches="tight")

### Best model statistics

n_preds = sum(bool_target_na)
in_sample_len = len(y_time_series)
tbats_season_comp = best_tbats_model.params.components.seasonal_periods

tbat_estimator = TBATS(seasonal_periods=tbats_season_comp)
full_tbats_model = tbat_estimator.fit(y_time_series)
all_in_sample_pred = full_tbats_model.y_hat
oos_pred, confid_info = full_tbats_model.forecast(n_preds,
                                                  confidence_level=0.95)
oos_pred[oos_pred < 0] = 0

fig, ax = plt.subplots(figsize=(20, 10))
y_time_series.plot(ax=ax, style="o", label="Actual")
ax.plot(all_in_sample_pred, color="r", linestyle="--",
        label="In Sample Predictions")
ax.plot(np.arange(in_sample_len, in_sample_len + n_preds),
        oos_pred, color="g", linestyle="-",
        label="Out of Sample Predictions")
ax.fill_between(np.arange(in_sample_len, in_sample_len + n_preds),
                confid_info["lower_bound"],
                confid_info["upper_bound"],
                color="g", alpha=0.2, label="95% Confidence Intervals")
ax.legend(loc="upper right", prop={"size": 20})
ax.tick_params(axis="both", labelsize=16)
fig.tight_layout()
fig.savefig(r"{}/iq_tbats_performance_final.png".format(output_path),
            bbox_inches="tight")

iq_pred = pd.DataFrame({
    "total_cases": np.round(oos_pred),
    "city": "iq",
    "year": data.loc[bool_target_na, "year"],
    "weekofyear": data.loc[bool_target_na,"weekofyear"],
    })

#%% Combining Predictions

all_preds = pd.concat([sj_pred, iq_pred])
all_preds.loc[:, "total_cases"] = all_preds.loc[:, "total_cases"].astype(int)
right_column_order = all_preds.loc[:, ["city",
                                       "year",
                                       "weekofyear",
                                       "total_cases"]]
right_column_order.to_csv("{}/predictions".format(output_path), index=False)
