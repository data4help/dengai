#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 19:49:24 2020

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
from _config import features, model_dict, imputation_model
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

# %% Imputation class

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

processed_data = pd.concat(list(transformed_data_dict.values()), axis=0)
processed_data.to_csv("{}/preprocessed_data.csv".format(data_path),
                      index=False)
