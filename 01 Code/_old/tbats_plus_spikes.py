#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 19:49:24 2020

@author: paulmora
"""

# %% Preliminaries

### Packages
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import pickle

import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import STL

### Paths
main_path = r"/Users/paulmora/Documents/projects/dengai"
raw_path = r"{}/00 Raw".format(main_path)
code_path = r"{}/01 Code".format(main_path)
data_path = r"{}/02 Data".format(main_path)
output_path = r"{}/03 Output".format(main_path)

os.chdir(code_path)
from _config import features, imputation_model
from _functions import model_train

### Loading the data
preprocessed_data = pd.read_csv(r"{}/preprocessed_data.csv".format(data_path))
data_dict = {}
cities = set(preprocessed_data.loc[:, "city"])
for city in cities:
    bool_city = preprocessed_data.loc[:, "city"] == city
    subset_data = preprocessed_data.loc[bool_city, :].reset_index(drop=True)
    data_dict[city] = subset_data

# %% Defining and predicting spikes




def acf_plots(y, max_lags):
    fig, axs = plt.subplots(figsize=(20, 10))
    sm.graphics.tsa.plot_acf(y.values.squeeze(),
                             lags=max_lags, ax=axs, missing="drop")
    axs.set_title("Autocorrelation Plot", fontsize=18)
    axs.tick_params(axis="both", labelsize=16)
    fig.tight_layout()
    fig.savefig(r"{}/{}_autocorrelation_function.png".format(output_path,
                                                             city_name),


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


def stl_decomposing(period):

res = STL(y, period=period).fit()
fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(15,8))
res.trend.plot(ax=ax1)
res.resid.plot(ax=ax2)
res.seasonal.plot(ax=ax3)

y = data.loc[:, "total_cases"].dropna()

acf_plots(y, 52*4)


fig, axs = 
res.plot(ax=axs)
    adjusted_series = time_series - res.seasonal
    data.loc[:, col] = adjusted_series

data = data_dict["sj"]

lag = 200
threshold = 5
peak_dummy = finding_moving_peaks(y, lag, threshold)

scaled_y_series = data.loc[:, "total_cases"] / np.max(data.loc[:, "total_cases"])

fig, axs = plt.subplots(figsize=(20, 10))
axs.plot(scaled_y_series)
axs.plot(signals)
