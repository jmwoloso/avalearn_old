# Authors: Jason Wolosonovich <jmwoloso@asu.edu>
# License: BSD 3 clause

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import chi2


class ChiSquared(object):
    """
    Chi-Squared test to measure independence between two nominal or
    ordinal variables. Wraps sklearn.feature_selection chi2 while
    providing additional functionality, including accepting and
    returning a pandas dataframe.

    Null Hypothesis: variables are statistically independent

    Alternative Hypothesis: variables are statistically dependent

    Test Statistic: Chi-Squared, also returns p-value which represents
        the area above the statistic in the chi-squared distribution
        with (r-1)*(c-1) degrees of freedom.

    Caveats:
        The uniform expected frequencies should number at least a
        count of 5, otherwise some other method should be used.

    Sources:
        http://stats.stackexchange.com/questions/108007/correlations-
            with-categorical-variables

    """
    def __init__(self):
        pass

    def fit(self, dframe):
        return None

    def transform(self, dframe):
        return None

    def fit_transform(self, dframe):
        return None


class CramersV(object):
    """
    Class to calculate Cramer's V-a measure of the strength of the
    association between two nominal variables-an extension of the
    Chi-Squared test of independence for two nominal variables.

    Sources:
        http://stats.stackexchange.com/questions/108007/correlations-
            with-categorical-variables

        http://datascience.stackexchange.com/questions/893/how-to-get-
            correlation-between-two-categorical-variable-and-a-
            categorical-variab
    """
    def __init__(self):
        pass

    def fit(self, dframe):

        return None

    def transform(self, dframe):
        return None

    def fit_transform(self, dframe):
        return None


class OneWayANOVA(object):
    """
    Class to measure the independence between nominal
    independent variables and a classification target using One-Way
    ANOVA. Wraps sklearn.feature_selection f_classif while
    providing additional functionality, including accepting and
    returning a pandas dataframe.

    Source:
        http://stats.stackexchange.com/questions/108007/correlations-
            with-categorical-variables

        http://datascience.stackexchange.com/questions/893/how-to-get-
            correlation-between-two-categorical-variable-and-a-
            categorical-variab
    """
    def __init__(self):
        pass

    def fit(self, dframe):
        return None

    def transform(self, dframe):
        return None

    def fit_transform(self, dframe):
        return None


class LoglinearAnalysis(object):
    """
    Class to perform Loglinear Analysis on multiple (more than two)
    nominal variables. For strictly two such variables,
    use stats.ChiSquared instead. Wraps sklearn.linear_model
    LogisticRegression using `multi-class' == 'multinomial' and
    `solver` == 'lbfgs'

    This test will determine the associations between all possible
    combinations of each level of the nominal variables and is
    appropriate for determining whether to create interaction
    variables or new features that incorporate rules for a decision
    tree.

    Source:
        laerd.com statistical test selector.

        http://stackoverflow.com/questions/33248791/using-scikit
            -learn-to-training-an-nlp-log-linear-model-for-ner
    """
    # TODO: LogLinearAnalysis: Consider using SBS or SBFS from mlxtend
    # TODO: LogLinearAnalysis: like the Laerd example does.
    def __init__(self):
        pass

    def fit(self, dframe):
        logistic = LogisticRegression(multi_class='multinomial',
                                      solver='lbfgs')
        return None

    def transform(self, dframe):
        return None

    def fit_transform(self, dframe):
        return None