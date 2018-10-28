#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    File name : predictmixed_class.py
    Description : Class for predicting response for mixed models.
    Author : Mathilde DUVERGER, Marie IVCHTCHENKO
    Date created : 2018/05/04
    Python Version : 3.6
"""

import numpy as np


class PredictMixed(object):

    """
    Summary
    -------
    Predict the response for a given group within a linear mixed model.

    Parameters
    ----------
    model : 'statsmodels.regression.mixed_linear_model.MixedLMResultsWrapper'
        Model from statsmodels package that was previously fit on
        the whole dataset.

    Note
    ----
    There will be one instance of PredictMixed per group.
    """

    def __init__(self, model):

        self.model = model
        self.mu = None

    def fit_re(self, Y, X_fe, X_re):
        """
        Summary
        -------
        Compute the maximum a posteriori estimator of the random effects
        for a group.

        Parameters
        ----------
        Y : 'pandas DataFrame'
            DataFrame of the response variable.
        X_fe : 'pandas DataFrame'
            DataFrame of the fixed effects covariates.
        X_re : 'pandas DataFrame'
            DataFrame of the random effects covariates.
        """

        sigma2 = self.model.scale
        Omega = self.model.cov_re
        iO = np.linalg.inv(Omega)

        fixef = self.model.predict(X_fe)
        res = Y - fixef
        A = X_re.values
        A = A.reshape((A.shape[0], -1))

        Gamma = np.linalg.inv(np.matmul(np.transpose(A), A)/sigma2 + iO)
        self.mu = np.matmul(np.matmul(Gamma, np.transpose(A)), res)/sigma2

    def predict_re(self, X_fe, X_re):
        """
        Summary
        -------
        Predict the response variable for the group and the given inputs.

        Parameters
        ----------
        X_fe : 'pandas DataFrame'
            DataFrame of the fixed effects covariates.
        X_re : 'pandas DataFrame'
            DataFrame of the random effects covariates.

        Returns
        -------
        prediction : 'numpy.ndarray'
            Array of predictions.

        Note
        ----
        The function fit_re should be called before using this function.
        """

        if any(self.mu) is None:
            raise AttributeError('You need to call the function fit_re to fit'
                                 + 'before predicting.')


        fixef = self.model.predict(X_fe)
        A = X_re.values
        A = A.reshape((A.shape[0], -1))
        prediction = np.array(fixef + np.matmul(A, self.mu))

        return prediction

    def predict_fe(self, X_fe):
        """
        Summary
        -------
        Predict the response variable for the group and the given inputs.

        Parameters
        ----------
        X_fe : 'pandas DataFrame'
            DataFrame of the fixed effects covariates.

        Returns
        -------
        prediction : 'numpy.ndarray'
            Array of predictions.
        """

        fixef = self.model.predict(X_fe)
        prediction = np.array(fixef)

        return prediction
