# Authors: Jason Wolosonovich
# License: BSD 3 clause


class CleanNumeric(object):
    """
    Class to clean numeric features similar to the `vtreat` R package
    created by Nina Zumel and John Mount (Win-Vector, LLC).

    Numeric features with missing values are imputed using the mean
    of the feature. This may not always be appropriate however. If
    the values are not missing at random, but rather due to some
    systematic process, imputing with the mean is incorrect.

    To account for this, the class will also create an indicator column
    for each numeric feature and place a 1 in the rows where the
    value was missing and 0 otherwise, which may allow downstream
    estimators to use the feature correctly, assuming the indicator
    and/or original feature survives any feature selection methods.

    """