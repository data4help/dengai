# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 17:03:24 2020

@author: PaulM
"""

# %% Packages

from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import copy

# %% Functions

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
