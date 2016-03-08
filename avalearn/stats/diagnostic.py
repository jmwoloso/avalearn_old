# Authors: Jason Wolosonovich <jmwoloso@asu.edu>
# License: BSD 3 clause
"""
diagnostic.py:  Various diagnostics used to check if certain
               functions and classes from avalearn.stats.test can be
               used appropriately on one or more dataframes.
"""

import numpy as np
import pandas as pd
from collections import OrderedDict
import json
from avalearn.utilities.functions import rounder

class CheckChiSquareRequirements(OrderedDict):
    # TODO: Need to calculate expected values for each in order to
    # test requirement #2 accurately
    """
    Class used to verify that the supplied categorical features
    (2 or more) meet the requirements for the Chi-square
    goodness-of-fit test, which are:

        1. Fewer than 1/5 of the categories being compared have fewer
           than 5 expected observations for each level, respectively.

        2. At least one observation is expected in every category
        (i.e. no expected values of 0)

        3. Sampling method is simple random

    Regarding requirement #3, often we get datasets with no
    information on the method  used to collect the samples (e.g.
    Kaggle competitions), therefore, this check may not be entirely
    reliable. Consider it one more data point that may provide
    clarity on which methods should/could be used for a given dataset.

    Regarding requirement #2: This refers to the EXPECTED values

    Parameters
    ----------
    train_dframe: pandas dataframe; required.
        Contains the training data.

    test_dframe: pandas dataframe; required.
        Contains the test data.

    categorical_columns: str, list or pandas.core.index.Index object; required.
        The categorical columns to analyze.

    Returns
    -------
    collections.OrderedDict object containing columns as the
    top-level keys.

    The first two keys represent the overall result
    for satisfying the two requirements that are being evaluated and
    their corresponding values are in ['PASS', 'FAIL'].

    The value of the third key holds a dictionary where the keys are
    in ['train', 'test'] and the value for each is a list containing
    any columns in that failed evaluation of the first requirement.

    The third key is there for convenience
    (examing the failing columns) in case the first requirement is
    not satisfied, but can also be useful even when the first
    requirement is satisfied.

    Each remaining key corresponds to the columns. These keys contain
    dictionaries where the keys are in ['train', 'test'] and the
    values are in ['PASS', ('FAIL', [])] with the failing category
    levels for that feature populating the list inside the tuple.

    Sources:
        https://en.wikipedia.org/wiki/Pearson's_chi-squared_test
        https://statsmethods.wordpress.com/2013/05/29/chi-square-goodness-of-fit-test/
        http://stattrek.com/chi-square-test/goodness-of-fit.aspx?Tutorial=AP
    """
    # TODO: CheckChiSquareRequirements: Input validation
    # TODO: CheckChiSquareRequirements: Tests
    # TODO: CheckChiSquareRequirements: Examples
    # TODO: CheckChiSquareRequirements: Print json??
    def __init__(self, categorical_columns=None):
        super(CheckChiSquareRequirements, self).__init__()
        # type check
        if not isinstance(categorical_columns, pd.Index):
            self.columns = pd.Index(categorical_columns)
        else:
            self.columns = categorical_columns
        self.column_count = len(self.columns)
        self.failing_count = 0
        self.critical_value = rounder(self.column_count * 0.20)
        self['Requirement_1'] = 'TBD'
        self['Requirement_2'] = 'TBD'
        self['Requirement_1_Failures'] = OrderedDict()
        self['Requirement_1_Failures']['train'] = list()
        self['Requirement_1_Failures']['test'] = list()

    def fit(self, train_dframe, test_dframe):
        """
        Create new dataframes with only the feature categories.
        """
        # subset the dataframes; replace missing
        self.__train = train_dframe.loc[:, self.columns].fillna(
            value='MISSING').copy()
        self.__test = test_dframe.loc[:, self.columns].fillna(
            value='MISSING').copy()
        return self

    def transform(self):
        """
        Analyze the cateogirical features and assess whether they
        pass the check.
        """
        # TODO: Check if the instance has been fitted

        # check requirements above
        for column in self.columns:
            # evaluate requirement #1
            self._train_unique_counts = self.__train.loc[:,
                               column].value_counts()
            self._test_unique_counts = self.__test.loc[:,
                                       column].value_counts()

            if self._train_unique_counts.min() < 5 \
            or self._test_unique_counts.min() < 5:
                self.failing_count += 1

            if self._train_unique_counts.min() < 5:
                self['Requirement_1_Failures']['train'].append(column)

            if self._train_unique_counts.min() < 5:
                self['Requirement_1_Failures']['test'].append(column)

            # evaluate requirement #2
            self._train_unique = list(set(self.__train.loc[:,
                                          column].unique()))
            self._test_unique = list(set(self.__test.loc[:,
                                         column].unique()))
            self._train_diff = list(np.setdiff1d(self._train_unique,
                                                 self._test_unique))
            self._test_diff = list(np.setdiff1d(self._test_unique,
                                                self._train_unique))

            self[column] = OrderedDict()
            self[column]['train'] = None
            self[column]['test'] = None

            if len(self._train_diff) == 0 and len(self._test_diff) == 0:
                self[column]['train'] = 'PASS'
                self[column]['test'] = 'PASS'

            if len(self._train_diff) != 0 \
                    and len(self._test_diff) == 0:
                self[column]['train'] = ['FAIL',
                                         self._train_diff]
                self[column]['test'] = 'PASS'

            if len(self._test_diff) != 0 \
                    and len(self._train_diff) == 0:
                self[column]['train'] = 'PASS'
                self[column]['test'] = ['FAIL',
                                        self._test_diff]

            if len(self._train_diff) != 0 \
                and len(self._test_diff) != 0:
                self[column]['train'] = ['FAIL',
                                         self._train_diff]
                self[column]['test'] = ['FAIL',
                                        self._test_diff]

        self['Requirement_2'] = 'PASS'
        return self

    def fit_transform(self, train_dframe, test_dframe):
        """
        Creates the subset of original dataframes and analyzes the
        categorical features, assessing whether they pass or not
        """
        self.fit(train_dframe, test_dframe)
        self.transform()
        return self


def check_simple_unimodal(dframe=None, column=None):
    """
    Function that provides a simple test for unimodal distribution of a
    given feature.

    This function implements two common methods of checking for
    multi-modal distribution (see sources for details).

    Any distribution with a single global maximum is, by definition,
    strictly unimodal, though areas corresponding to local maxima
    within the distribution are often referred to as modes; indeed
    pandas implements pd.DataFrame.mode() in a similar manner.

    Therefore, a dataframe containing any feature that returns
    multiple modes (via pd.DataFrame.mode()) should be visually
    inspected using avalearn.utilities.make_numeric_distplots as well as
    tested using this function.

    If results fail or are inconclusive, the next step should be a
    gaussian mixture model and/or kdeplot (seaborn) to visually
    inspect for multi-modality.

    Sources:
        https://en.wikipedia.org/wiki/Mode_%28statistics%29

        https://en.wikipedia.org/wiki/Unimodality

    Parameters
    ----------
    dframe: pandas dataframe object.

    column: str; required.
        The column to perform the simple unimodal test on.

    """
    # TODO: Input validation
    df = dframe.loc[:, column].copy()

    # test results
    test_results = []

    mean = df.mean()
    median = df.median()
    modes = df.mode()
    std = df.std()
    bound = (3./5)**(1./2)

    # unimodal test for median/mean
    unimodal = np.abs(median - mean) / std <= bound

    test_results.append(unimodal)

    # different bound exists for comparing median and mode(s)
    bound = 3**(1./2)
    # test each mode as well
    for mode in modes:
        unimodal = np.abs(median - mode) / std <= bound
        test_results.append(unimodal)

    # produce final results

    # if all tests return 'True', we have a unimodal distribution
    if np.sum(test_results) / len(test_results) == 1:
        unimodal = True

    # if all tests return 'False', distribution is bimodal
    if np.sum(test_results) / len(test_results) == 0:
        unimodal = False

    # if any of the tests return 'False', results are inconclusive
    if np.sum(test_results) / len(test_results) not in [0, 1]:
        unimodal = "INCONCLUSIVE: {0} Failed."\
            .format(len(test_results) - np.sum(test_results))

    return unimodal

















