# Authors: Jason Wolosonovich
# License: BSD 3 clause
"""
summary.py: Convenience functions for calculating statistical measures.
"""

import numpy as np
import pandas as pd


def cohens_d(dframe=None, feature=None, target=None):
    """
    Calculates Cohen's D statistic which compares the difference
    between groups to the variability within groups. This
    implementation is useful for comparing numeric values to
    the mean of the `target` but is also a method of ImpactCoder when
    encoding the levels of a categorical variable.

    Source:
        Think Stats, 2nd Edition by Allen B. Downey

    Parameters
    ----------
    dframe: pandas dataframe; required
        The dataframe containing the columns to be analyzed.

    feature: str; required
        Column name in a pandas dataframe used to compute the D stat.

    target: str; required
        Target column used to determine the effect size of `feature`



    """
    # find the difference of means
    diff = dframe.loc[:, feature].mean() - dframe.loc[:, target].mean()

    # compute the variance for each group
    feature_var = dframe.loc[:, feature].var()
    target_var = dframe.loc[:, target].var()

    # compute the lengths
    feature_len = dframe.shape[0]
    target_len = dframe.shape[0]

    # compute the pooled variance
    pooled_var = (feature_len * feature_var +
                  target_len * target_var)/\
                 (feature_len + target_len)

    # compute Cohen's D
    d = diff / np.sqrt(pooled_var)

    return d


def cdf(dframe=None):
    """
    Calculates the 
    """
