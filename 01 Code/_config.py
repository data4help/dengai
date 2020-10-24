# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 16:57:47 2020

@author: PaulM
"""
# %% Packages
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn import linear_model

# %%


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

features_w_target = features + ["total_cases"]

model_dict = {
    # Gradient Boosting Regressor
    "xgb": {
        "model": GradientBoostingRegressor(random_state=28),
        "param": {
            "n_estimators": [1_000],
            "learning_rate": [0.1, 0.2, 0.3],
            "min_samples_split": [2, 5, 10, 15, 100],
            "min_samples_leaf": [1, 2, 5, 10]
        },
    },
    # RandomForest Regressor
    "rfr": {
        "model": RandomForestRegressor(random_state=28),
        "param": {
            "n_estimators": [100],
            "max_depth": [int(x) for x in np.linspace(10, 110, num=10)],
            "min_samples_split": [2, 5, 10, 15, 100],
            "min_samples_leaf": [1, 2, 5, 10]
            }
        },
    # Ridge Regression
    "ridge": {
        "model": linear_model.Ridge(random_state=28),
        "param": {
            "alpha": [1e-8, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 3, 5, 10]
            }
        }
}


imputation_model = {
        "model": RandomForestRegressor(random_state=28),
        "param": {
            "n_estimators": [100],
            "max_depth": [int(x) for x in
                          np.linspace(10, 110, num=10)],
            "min_samples_split": [2, 5, 10, 15, 100],
            "min_samples_leaf": [1, 2, 5, 10]
            }
        }


lasso_model_dict = {
    # Lasso Regression
    "model": linear_model.Lasso(random_state=28, fit_intercept=True),
    "param": {
        "alpha": [1e-8, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
        }
    }

