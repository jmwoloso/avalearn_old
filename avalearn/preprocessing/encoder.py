# Authors: Jason Wolosonovich <jmwoloso@asu.edu>
# License: BSD 3 clause

import numpy as np
import pandas as pd
from pandas import get_dummies
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB

__all__ = [
    'MultiColumnLabelEncoder',
    'ImpactEncoder'
]


class MultiColumnLabelEncoder(LabelEncoder):
    """
    Wraps sklearn LabelEncoder functionality for use on multiple
    columns of a pandas dataframe.

    """
    # TODO: MultiColumnLabelEncoder: docstrings
    # TODO: MultiColumnLabelEncoder: input validation
    # TODO: MultiColumnLabelEncoder: tests
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, dframe):
        """
        Fit label encoder to pandas columns.

        Access individual column classes via indexig `self.all_classes_`

        Access individual column encoders via indexing
        `self.all_encoders_`
        """
        df = dframe.copy()
        # if columns are provided, iterate through and get `classes_`
        if self.columns is not None:
            # ndarray to hold LabelEncoder().classes_ for each
            # column; should match the shape of specified `columns`
            self.all_classes_ = np.ndarray(shape=self.columns.shape,
                                           dtype=object)
            self.all_encoders_ = np.ndarray(shape=self.columns.shape,
                                            dtype=object)
            for idx, column in enumerate(self.columns):
                # fit LabelEncoder to get `classes_` for the column
                le = LabelEncoder()
                le.fit(df.loc[:, column].values)
                # append the `classes_` to our ndarray container
                self.all_classes_[idx] = (column,
                                          np.array(le.classes_.tolist(),
                                                  dtype=object))
                # append this column's encoder
                self.all_encoders_[idx] = le
        else:
            # no columns specified; assume all are to be encoded
            self.columns = df.iloc[:, :].columns
            self.all_classes_ = np.ndarray(shape=self.columns.shape,
                                           dtype=object)
            for idx, column in enumerate(self.columns):
                le = LabelEncoder()
                le.fit(df.loc[:, column].values)
                self.all_classes_[idx] = (column,
                                          np.array(le.classes_.tolist(),
                                                  dtype=object))
                self.all_encoders_[idx] = le
        return self

    def fit_transform(self, dframe):
        """
        Fit label encoder and return encoded labels.

        Access individual column classes via indexing
        `self.all_classes_`

        Access individual column encoders via indexing
        `self.all_encoders_`

        Access individual column encoded labels via indexing
        `self.all_labels_`
        """
        df = dframe.copy()
        # if columns are provided, iterate through and get `classes_`
        if self.columns is not None:
            # ndarray to hold LabelEncoder().classes_ for each
            # column; should match the shape of specified `columns`
            self.all_classes_ = np.ndarray(shape=self.columns.shape,
                                           dtype=object)
            self.all_encoders_ = np.ndarray(shape=self.columns.shape,
                                            dtype=object)
            self.all_labels_ = np.ndarray(shape=self.columns.shape,
                                          dtype=object)
            for idx, column in enumerate(self.columns):
                # instantiate LabelEncoder
                le = LabelEncoder()
                # fit and transform labels in the column
                df.loc[:, column] =\
                    le.fit_transform(df.loc[:, column].values)
                # append the `classes_` to our ndarray container
                self.all_classes_[idx] = (column,
                                          np.array(le.classes_.tolist(),
                                                  dtype=object))
                self.all_encoders_[idx] = le
                self.all_labels_[idx] = le
        else:
            # no columns specified; assume all are to be encoded
            self.columns = df.iloc[:, :].columns
            self.all_classes_ = np.ndarray(shape=self.columns.shape,
                                           dtype=object)
            for idx, column in enumerate(self.columns):
                le = LabelEncoder()
                df.loc[:, column] = le.fit_transform(
                        df.loc[:, column].values)
                self.all_classes_[idx] = (column,
                                          np.array(le.classes_.tolist(),
                                                  dtype=object))
                self.all_encoders_[idx] = le
        return df

    def transform(self, dframe):
        """
        Transform labels to normalized encoding.
        """
        df = dframe.copy()
        if self.columns is not None:
            for idx, column in enumerate(self.columns):
                df.loc[:, column] = self.all_encoders_[
                    idx].transform(df.loc[:, column].values)
        else:
            self.columns = df.iloc[:, :].columns
            for idx, column in enumerate(self.columns):
                df.loc[:, column] = self.all_encoders_[idx]\
                    .transform(df.loc[:, column].values)
        return df

    def inverse_transform(self, dframe):
        """
        Transform labels back to original encoding.
        """
        df = dframe.copy()
        if self.columns is not None:
            for idx, column in enumerate(self.columns):
                df.loc[:, column] = self.all_encoders_[idx]\
                    .inverse_transform(df.loc[:, column].values)
        else:
            self.columns = df.iloc[:, :].columns
            for idx, column in enumerate(self.columns):
                df.loc[:, column] = self.all_encoders_[idx]\
                    .inverse_transform(df.loc[:, column].values)
        return df


class ImpactEncoder(object):
    """
    Impact coding class for categorical features in a Pandas
    Dataframe based on the `vtreat` R package developed by Nina
    Zumel and John Mount (Win-Vector/ Win-Vector blog).

    Sources:
        http://www.win-vector.com/blog/2012/07/modeling-trick-
            impact-coding-of-categorical-variables-with-many-levels/

        http://www.win-vector.com/blog/2014/08/vtreat-designing-
            a-package-for-variable-treatment/

    The concept is to make use of categorical features that have many
    different levels by fitting a Bayesian model
    (sklearn.naive_bayes.MultinomialNB) for each level of each
    categorical feature and then replacing the level name with the
    difference between probability of that category level and the Grand
    Mean probability of `target_features`.

    Parameters
    ----------
    target_features: list, pandas.core.index.Index object

        Default: None

        List of the target feature column names.

    categorical_features: list, pandas.core.index.Index object

        Default: None

        List of the categorical feature column names.

    convert_to_categorical: list, pandas.core.index.Index object

        Default: None, optional

        List of additional columns to convert to categorical columns
        which will then be impact encoded as well.

    min_categorical_loss: float in range [0,1]
        minimum percentage of missing data allowed for any level of a
        categorical feature before conversion switches from binary
        encoding to impact encoding

    Attributes
    ----------


    Examples
    --------
    """
    # TODO: ImpactCoding: docstrings
    # TODO: ImpactCoding: input validation
    # TODO: ImpactCoding: tests
    # TODO: ImpactCoding: add multi-class support
    # TODO: ImpactCoding: add Cohen's D method
    def __init__(self, target_features=None, categorical_features=None,
                 convert_to_categorical=None, min_categorical_loss=0.02,
                 max_data_loss=0.04, alpha=1.0, fit_prior=True,
                 class_prior=None):
        self.targets = target_features
        self.categories = categorical_features
        self.to_convert = convert_to_categorical
        self.min_loss = min_categorical_loss
        self.max_loss = max_data_loss
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior

    def fit(self, dframe):
        """
        Fit an sklearn.naive_bayes MultinomialNB model to get the
        category-level conditional probabilities for the target class.

        Parameters
        ----------
        dframe: pandas dataframe

        Returns
        -------
        self: returns an instance of self (MultinomialNB model)
        """
        mnb = MultinomialNB(alpha=self.alpha, fit_prior=self.fit_prior,
                            class_prior=self.class_prior)
        return None

    def transform(self, X):
        """
        Transform based on priors, grand mean discovered by `fit`.
        """
        return None

    def fit_transform(self, X, y):
        """
        Fit to the data then transform to impact encodings based on
        priors, grand mean discovered.
        """
        return None

    def inverse_transform(self, X, y):
        """
        Reverse the impact encodings back to the original values.
        """
        return None


class DummyEncoder(object):
    """
    Class implementing pandas.get_dummies with some added
    functionality such as preventing collinearity by only creating
    N-1 indicator variables for categorical columns (if specified).

    In most machine learning applications, this is not of concern
    since accurate prediction, rather than feature interpretation,
    is of primary importance. Never the less, this option is provided as
    a convenience should the need arise.

    Additionally, incorporates features similar to those in the
    `vtreat` R package created by Nina Zumel and John Mount (
    Win-Vector, LLC) which removes indicators (setting them equal to
    0) in which a given category-level indicator is "on" (value == 1)
    <= `min_categorical_loss` percent of the time.

    Also keeps a count of what percentage of the total data for a
    given categorical variable ends up being lost (value set == 0)
    and when that total >= `max_total_loss`, returns a warning that
    the user should consider using ImpactEncoder on the specified
    columns before attempting to created indicators. This step and
    the prior step happen during the call to fit.


    Parameters
    ----------

    categorical_features: list, pandas.core.index.Index object

        default: None; optional; do not use with
        `allow_collinearity` == True.

        List specifying the categorical features in the dataframe.
        Pandas detects these automatically and knows what to do,
        however, if `allow_collinearity` == False, then the
        `categorical_features` must be supplied so that one indicator
        per feature may be removed.

    target_features: list, pandas.core.index.Index object

        default: None; optional; do not use with
        `allow_collinearity` == True.

        List specifying the target feature(s) in the dataframe. Only
        required if `allow_collinearity` == False so that none of the
        target features will have columns removed in the case of
        multi-class targets.

    allow_collinearity:  bool

        default: True; required.

        Boolean value specifying whether or not to allow
        collinearity among a feature's indicators by creating N
        indicator variables. If False, N-1 indicator variables will
        be returned for each categorical feature.

    convert_to_categorical: list, pandas.core.index.Index object

        default: None; optional.

        Additional columns to convert to categorical and then create
        indicators for. Useful for columns that contain numeric
        values representing category levels.

        Use: a column containing years (e.g. 2012, 2013, etc.)

    prefix_sep: str

        default: '='; required.

        Used by pandas.get_dummies as the `prefix_sep` item when
        creating indicator column names.

        Usage: Suppose a category ('CategoryField') with 3 distinct
               levels, ['A', 'B', 'C']. Using the default value ('=')
               results in new indicator columns ['CategoryField=A',
               'CategoryField=B', 'CategoryField=C']

    min_categorical_loss: float in range [0,1] or None.

        Default: 0.02; optional.

        Float representing the percentage of any single level in a
        categorical feature that can be missing before coding all of
        those indicators == 0. Useful for categorical features with
        many levels where only a few levels represent a substantial
        portion of the category levels,

    max_total_loss: float in range [0,1] or None.

        Default: 0.04, optional.

        Float representing to total information loss allowed for a
        given categorical feature before it is suggested to use
        ImpactEncoder before attempting to make indicators.

    """
    def __init__(self, allow_collinearity=True):
        """
        Instantiate the class and pass arguments.
        """
        self.collinearity = allow_collinearity

    def fit(self, dframe):
        return self

    def transform(self, dframe):
        return None

    def fit_transform(self, dframe):
        return None


