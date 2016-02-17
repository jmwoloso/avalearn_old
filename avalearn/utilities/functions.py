# Authors: Jason Wolosonovich
# License: BSD 3 clause
"""
functions.py: A collection of convenience functions for various tasks

"""

from time import time
import numpy as np
import pandas as pd
from scipy import stats, integrate
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from itertools import combinations
import pickle
import statsmodels.api as sm
from collections import OrderedDict
from sklearn.cross_validation import train_test_split




def make_numeric_dist_plots(dframe=None, skip_columns=None):
    """
    Convenience function for making and saving Seaborn plots for
    numeric features.

    NOTE: This function saves the figures in the current directory

    Parameters
    ----------
    dframe: pandas dataframe
        The dataframe to make plots for

    skip_columns: list or pandas.core.index.Index object
        The columns that should be skipped when creating the plots.

    """
    # TODO: Input validation
    print("Creating dist plots.")

    # select desired color scheme
    sns.set_style("darkgrid")

    # get the original backend used
    orig_backend = plt.get_backend()

    # change backend to prevent windows from popping up for each loop
    if orig_backend != 'agg':
        plt.switch_backend('Agg')

    # find out if interactive plotting is on, turn off if so
    plt_state = plt.isinteractive()
    if plt_state:
        plt.ioff()

    # drop the `skip_columns`
    df = dframe.copy()
    if skip_columns:
        df = df.drop(skip_columns,
                             axis=1)

    # get only numeric columns and drop the rest, backup in case user
    # forgot to drop or exclude non-numeric columns
    numeric_columns = df.iloc[:, :].select_dtypes(include=['int',
                                                           'float']).columns

    # create the plots for each remaining column
    for column in numeric_columns:
        sns_plot = sns.distplot(df.loc[:, column].dropna().values,
                                fit=stats.norm,
                                axlabel=column)
        fig = sns_plot.get_figure()
        fig.savefig(column + '_dist.png')
        plt.cla()
        plt.close()
        plt.close('all')


    # change the plotting back to the original if necessary
    if plt_state:
        plt.ion()

    print("Successfully created and saved distribution plots.")


def make_box_plots(dframe=None, target_columns=None, skip_columns=None):
    """
    Convenience function for making and saving Seaborn plots for
    categorical features.

    NOTE: This function saves the figures in the current directory

    Parameters
    ----------
    dframe: pandas dataframe; required
        The dataframe to be plotted.

    target_columns: list or pandas.core.index.Index object; required
        The target column against which each categorical feature will be
        plotted.

    skip_columns: list or pandas.core.index.Index object; optional
        The columns that should be skipped when creating the plots.

    """
    # TODO: Make this function get dataframe into tidy format
    # TODO: Input validation

    print("Creating box plots.")

    # set to white grid for ease of readability with these types of
    # plots
    sns.set_style("whitegrid")

    # get the original backend used
    orig_backend = plt.get_backend()

    # change backend to prevent windows from popping up for each loop
    if orig_backend != 'agg':
        plt.switch_backend('Agg')

    # find out if interactive plotting is on, turn off if so
    plt_state = plt.isinteractive()
    if plt_state:
        plt.ioff()

    # drop the `skip_columns`
    df = dframe.copy()
    if skip_columns:
        df = df.drop(skip_columns,
                             axis=1)

    # get only categorical columns and drop the rest, backup in case
    # user forgot to drop or exclude non-numeric columns, etc.
    categorical_columns = df.iloc[:, :]\
        .select_dtypes(exclude=['int',
                                'float',
                                # 'timedelta[ns]' # doesn't work
                                'datetime64']).columns

    # create the plots for each remaining column vs. all other columns
    for target in target_columns:
        for column in categorical_columns:
            sns_plot = sns.boxplot(x=column,
                                   y=target,
                                   data=df)
                                   # order=sorted(df.loc[:,
                                   #              column].unique()))
            fig = sns_plot.get_figure()
            fig.savefig(column + '_' + target + '_box.png')
            plt.cla()
            plt.close()
            plt.close('all')

    # change the plotting back to the original if necessary
    if plt_state:
        plt.ion()

    print("Successfully created and saved box plots.")


def make_numeric_hexbin_plots(dframe=None, target_columns=None,
                              skip_columns=None, sample_size=None,
                              stratify=False):
    """
    Convenience function for making and saving Seaborn bivariate
    scatterplots for numeric features.

    NOTE: This function saves the figures in the current directory

    Parameters
    ----------
    dframe: pandas dataframe
        The dataframe to make plots with.

    target_columns: list or pandas.core.index.Index object; optional.
        The target column is only needed if `sample_size` and
        `sample_method` are supplied. Needed for reconstructing the
        dataframe prior to creating the hexbin plots.

    skip_columns: list or pandas.core.index.Index object
        The columns that should be skipped when creating the plots.

    sample_size: float in range [0,1]; optional.
        Hexbin plots are more useful for reasonably sized samples;
        bigger hexbins results giving a clearer picture of the
        relationship between two variables (visually speaking).

    stratify: bool; optional.

        Default: False

        Whether to apply stratified sampling.

    """
    # TODO: Input validation
    # TODO: Test behavior for multi-class data; not sure how sklearn
    # TODO: handles `stratify` in the multi=class case.

    print("Creating hexbin plots.")

    # set to white
    sns.set_style("white")

    # get the original backend used
    orig_backend = plt.get_backend()

    # change backend to prevent windows from popping up for each loop
    if orig_backend != 'agg':
        plt.switch_backend('Agg')

    # find out if interactive plotting is on, turn off if so
    plt_state = plt.isinteractive()
    if plt_state:
        plt.ioff()

    # drop the `skip_columns`
    df = dframe.copy()
    if skip_columns:
        df = df.drop(skip_columns,
                             axis=1)

    # get only float columns and drop the rest, backup in case user
    # forgot to drop or exclude non-numeric columns
    float_columns = df.iloc[:, :].select_dtypes(include=[
        'float']).columns

    # generate samples
    if sample_size is not None:
        targets = df.loc[:, target_columns].values.copy()

        stratify_dict = {True: targets,
                         False: None}

        df = df.loc[:, float_columns]
        df_columns = df.iloc[:, :].columns
        X = df.iloc[:, :].values.copy()
        X_, _, y_, _ = train_test_split(
            X,
            targets,
            train_size=sample_size,
            stratify=stratify_dict[stratify]
        )
        df = pd.DataFrame(X_, columns=df_columns)
        df.loc[:, target_columns] = y_


    # create the plots for each remaining column
    for v1, v2 in combinations(float_columns, 2):
        try:
            # xmin = int(df3.loc[:, v1].min() - 0.25)
            # xmax = int(df3.loc[:, v1].max() + 0.25)
            # ymin = int(df3.loc[:, v2].min() - 0.25)
            # ymax = int(df3.loc[:, v2].max() + 0.25)
            sns_plot = sns.jointplot(x=v1,
                                     y=v2,
                                     data=df,
                                     kind="hex",
                                     color="#4CB391",
                                     dropna=True)
                                     # xlim=(-10, 20),
                                     # ylim=(-5, 25))

            sns_plot.savefig(v1 + '_' + v2 + '_hex.png')
            # not plt.cla()
            plt.clf()
            plt.close()
            plt.close('all')

        except MemoryError as e:
            print("Encountered a MemoryError for columns {0} and {"
                  "1}".format(v1, v2))
            continue

    # change the plotting back to the original if necessary
            if plt_state:
                plt.ion()
    print("Successfully created and saved hexbin plots.")


def make_count_bar_plots(dframe=None, skip_columns=None,
                         target_columns=None):
    """
    Convenience function for making and saving Seaborn bar plots for
    int (count?) features.

    NOTE: This function saves the figures in the current directory;
    not appropriate for int columns that are not truly counts,
    but worth looking at if the features are anonymous and have a
    reasonable number of values.

    Parameters
    ----------
    dframe: pandas dataframe; required
        The dataframe to be plotted.

    target_columns: list or pandas.core.index.Index object; required
        The target column against which each categorical feature will be
        plotted.

    skip_columns: list or pandas.core.index.Index object; optional
        The columns that should be skipped when creating the plots.

    """
    # TODO: Input validation

    print("Creating bar plots.")

    # set to white grid for ease of readability with these types of
    # plots
    sns.set_style("dark")

    # get the original backend used
    orig_backend = plt.get_backend()

    # change backend to prevent windows from popping up for each loop
    if orig_backend != 'agg':
        plt.switch_backend('Agg')

    # find out if interactive plotting is on, turn off if so
    plt_state = plt.isinteractive()
    if plt_state:
        plt.ioff()

    # drop the `skip_columns`
    df = dframe.copy()
    if skip_columns:
        df = df.drop(skip_columns,
                             axis=1)

    # get only categorical and int columns and drop the rest, backup in
    # case user forgot to drop or exclude non-numeric columns, etc.
    count_columns = df.iloc[:, :]\
        .select_dtypes(exclude=['float',
                                # 'timedelta[ns]' # doesn't work
                                'datetime64']).columns

    # drop targets
    count_columns = count_columns.drop(target_columns)


    # create the plots for each remaining column vs. all other columns
    for target in target_columns:
        for column in count_columns:
            sns_plot = sns.barplot(x=target,
                                   y=column,
                                   data=df,
                                   orient='h',
                                   order=sorted(set(df.loc[:,
                                                    column].unique())))

            fig = sns_plot.get_figure()
            fig.savefig(column + '_' + target + '_bar.png')
            plt.cla()
            plt.close()
            plt.close('all')


    # change the plotting back to the original if necessary
    if plt_state:
        plt.ion()

    print("Successfully created and saved bar plots.")


def make_count_heat_maps(dframe=None, target_columns=None,
                         skip_columns=None):
    """
    Convenience function for making and saving Seaborn heatmap plots for
    count (int?) and categorical features against targets.

    NOTE: This function saves the figures in the current directory;
    not appropriate for int columns that are not truly counts,
    but worth looking at if the features are anonymous and have a
    reasonable number of values.

    Parameters
    ----------
    dframe: pandas dataframe; required
        The dataframe to be plotted.

    target_columns: list or pandas.core.index.Index object; required
        The target column against which each categorical feature will be
        plotted.

    skip_columns: list or pandas.core.index.Index object; optional
        The columns that should be skipped when creating the plots.

    """
    # TODO: Input validation

    print("Creating heatmaps.")

    # set to white grid for ease of readability with these types of
    # plots
    sns.set_style("dark")

    # get the original backend used
    orig_backend = plt.get_backend()

    # change backend to prevent windows from popping up for each loop
    if orig_backend != 'agg':
        plt.switch_backend('Agg')

    # find out if interactive plotting is on, turn off if so
    plt_state = plt.isinteractive()
    if plt_state:
        plt.ioff()

    # drop the `skip_columns`
    df = dframe.copy()
    if skip_columns:
        df = df.drop(skip_columns,
                             axis=1)

    # get only categorical and int columns and drop the rest, backup in
    # case user forgot to drop or exclude non-numeric columns, etc.
    heat_columns = df.iloc[:, :]\
        .select_dtypes(exclude=['float',
                                # 'timedelta[ns]' # doesn't work
                                'datetime64']).columns

    # drop target
    heat_columns = heat_columns.drop(target_columns)

    # create the plots for each remaining column vs. all other columns
    for target in target_columns:
        for v1, v2 in combinations(heat_columns, 2):
            pivot = df.pivot_table(values=target,
                                   index=v1,
                                   columns=v2,
                                   aggfunc=np.mean)
            sns_plot = sns.heatmap(pivot,
                                   vmin=0,
                                   vmax=1,
                                   linewidths=0.5,
                                   robust=True,
                                   cmap='YlOrBr',
                                   annot=True)
            fig = sns_plot.get_figure()
            fig.savefig(v1 + '_' + v2 + '_heat.png')
            plt.cla()
            plt.close()
            plt.close('all')

    # change the plotting back to the original if necessary
    if plt_state:
        plt.ion()

    print("Successfully created and saved heat maps.")


def make_count_plots(dframe=None, target_columns=None,
                     skip_columns=None):
    """
    Convenience function for making and saving Seaborn count plots for
    count (int?) and categorical features against targets.

    NOTE: This function saves the figures in the current directory;
    not appropriate for int columns that are not truly counts,
    but worth looking at if the features are anonymous and have a
    reasonable number of values.

    Parameters
    ----------
    dframe: pandas dataframe; required
        The dataframe to be plotted.

    target_columns: list or pandas.core.index.Index object; required
        The target column against which each categorical feature will be
        plotted.

    skip_columns: list or pandas.core.index.Index object; optional
        The columns that should be skipped when creating the plots.

    """
    # TODO: Input validation

    print("Creating count plots.")

    # set to white grid for ease of readability with these types of
    # plots
    sns.set_style("darkgrid")

    # get the original backend used
    orig_backend = plt.get_backend()

    # change backend to prevent windows from popping up for each loop
    if orig_backend != 'agg':
        plt.switch_backend('Agg')

    # find out if interactive plotting is on, turn off if so
    plt_state = plt.isinteractive()
    if plt_state:
        plt.ioff()

    # drop the `skip_columns`
    df = dframe.copy()
    if skip_columns:
        df = df.drop(skip_columns,
                             axis=1)

    # get only categorical and int columns and drop the rest, backup in
    # case user forgot to drop or exclude non-numeric columns, etc.
    count_columns = df.iloc[:, :]\
        .select_dtypes(exclude=['float',
                                # 'timedelta[ns]' # doesn't work
                                'datetime64',
                                ]).columns

    # create the plots for each remaining column vs. all other columns
    for target in target_columns:
        for column in count_columns:
            sns_plot = sns.countplot(x=column,
                                     data=df,
                                     hue=target,
                                     palette='spectral')
            fig = sns_plot.get_figure()
            fig.savefig(column + '_' + target + '_count.png')
            plt.cla()
            plt.close()
            plt.close('all')

    # change the plotting back to the original if necessary
    if plt_state:
        plt.ion()

    print("Successfully created and saved count plots.")


def make_numeric_logistic_lm_plots(dframe=None, target_column=None,
                                   skip_columns=None, sample_size=None,
                                   stratify=False):
    """
    Convenience function for making and saving Seaborn logistic lm
    plots for numeric features against targets.

    NOTE: This function saves the figures in the current directory;
    not appropriate for int columns that are not truly counts,
    but worth looking at if the features are anonymous and have a
    reasonable number of values.

    Parameters
    ----------
    dframe: pandas dataframe; required
        The dataframe to be plotted.

    target_column: str, list, pandas.core.index.Index object; required.
        The target column against which each categorical feature will be
        plotted.

    skip_columns: list or pandas.core.index.Index object; optional
        The columns that should be skipped when creating the plots.

    sample_size: float in range [0,1]; optional.
        Hexbin plots are more useful for reasonably sized samples;
        bigger hexbins results giving a clearer picture of the
        relationship between two variables (visually speaking).

    stratify: bool; optional.

        Default: False

        Whether to apply stratified sampling.

    """
    # TODO: Input validation

    print("Creating logistic plots.")

    # set to white grid for ease of readability with these types of
    # plots
    sns.set_style("darkgrid")

    # get the original backend used
    orig_backend = plt.get_backend()

    # change backend to prevent windows from popping up for each loop
    if orig_backend != 'agg':
        plt.switch_backend('Agg')

    # find out if interactive plotting is on, turn off if so
    plt_state = plt.isinteractive()
    if plt_state:
        plt.ioff()

    # drop the `skip_columns`
    df = dframe.copy()
    if skip_columns:
        df = df.drop(skip_columns,
                             axis=1)

    # get only categorical columns and drop the rest, backup in case
    # user forgot to drop or exclude non-numeric columns, etc.
    float_columns = df.iloc[:, :].select_dtypes(include=[
        'float']).columns

    # generate samples
    if sample_size is not None:
        targets = df.loc[:, target_column].values.copy()

        stratify_dict = {True: targets,
                         False: None}

        df = df.loc[:, float_columns]
        df_columns = df.iloc[:, :].columns
        X = df.iloc[:, :].values.copy()
        X_, _, y_, _ = train_test_split(
            X,
            targets,
            train_size=sample_size,
            stratify=stratify_dict[stratify]
        )
        df = pd.DataFrame(X_, columns=df_columns)
        df.loc[:, target_column] = y_

    # create the plots for each remaining column vs. all other columns
    for column in float_columns:
        sns_plot = sns.lmplot(x=column,
                              y=target_column,
                              data=df,
                              logistic=True,
                              # lowess=True,
                              y_jitter=0.04,
                              ci=True)

        # fig = sns_plot.get_figure()
        sns_plot.savefig(column + '_' + target_column + '_logistic.png')
        plt.cla()
        plt.close()
        plt.close('all')

    # change the plotting back to the original if necessary
    if plt_state:
        plt.ion()

    print("Successfully created and saved logistic lm plots.")


def make_numeric_prob_plots(dframe=None, target_columns=None,
                            skip_columns=None):
    """
    Convenience function that uses Matplotlib via Scipy, to plot
    normality plots for all continuous features.

    Parameters
    ----------
    dframe: pandas dataframe object; required.

    skip_columns: list or pandas.core.index.Index object; optional
        The columns that should be skipped when creating the plots.

    """
    # TODO: Input validation

    print("Creating prob plots.")

    # quantile plot def
    def quantile_plot(x, **kwargs):
        """
        Source:
            https://stanford.edu/~mwaskom/software/seaborn/tutorial
            /axis_grids.html
        """
        qntls, xr = stats.probplot(x,
                                   fit=False)
        plt.scatter(xr, qntls, **kwargs)

    # set to white grid for ease of readability with these types of
    # plots
    sns.set_style("white")

    # get the original backend used
    orig_backend = plt.get_backend()

    # change backend to prevent windows from popping up for each loop
    if orig_backend != 'agg':
        plt.switch_backend('Agg')

    # find out if interactive plotting is on, turn off if so
    plt_state = plt.isinteractive()
    if plt_state:
        plt.ioff()

    # drop the `skip_columns`
    df = dframe.copy()
    if skip_columns:
        df = df.drop(skip_columns,
                             axis=1)

    # get only categorical columns and drop the rest, backup in case
    # user forgot to drop or exclude non-numeric columns, etc.
    float_columns = df.iloc[:, :].select_dtypes(include=[
        'float']).columns
    for target in target_columns:
        for column in float_columns:
            g = sns.FacetGrid(df,
                              col='target',
                              size=4)
            g.map(quantile_plot,
                  column,
                  color='chartreuse',
                  label="Probability Plot for {0} vs. {1}"
                  .format(column,
                          target))
            #fig = g.get_figure() # exclude this for FacetGrid
            g.savefig(column + '_' + target + '_prob.png')
            plt.cla()
            plt.close()
            plt.close('all')


def pickle_model(python_object=None, fname=None, mode='rwb'):
    """
    Convenience function for pickling a python object.

    Parameters
    ----------
    python_object: python object; required.
        Any python object that pickle can serialize.

    fname: str; required
        Name for the pickled object.

    mode: str; required.
        Default: 'wb' (write bytes - Python 3)
        Write mode for the pickle object
    """
    # TODO: Add functionality for changing the pickle write mode
    print("Pickling the object.")
    # pickle the object
    try:
        with open(fname, mode) as f:
             pickle.dump(python_object, f)
        print("Successfully pickled the object.")
    except IOError as e:
        print(e)
        print("Could not pickle the object. Something went wrong.")


def reindex_columns(dframe=None, columns=None, new_indices=None):
    """
    Reorders the columns of a dataframe as specified by
    `reorder_indices`. Values of `columns` should align with their
    respective values in `new_indices`.

    `dframe`: pandas dataframe.

    `columns`: list,pandas.core.index.Index, or numpy array; columns to
    reindex.

    `reorder_indices`: list of integers or numpy array; indices
    corresponding to where each column should be inserted during
    re-indexing.
    """
    print("Re-indexing columns.")
    try:
        df = dframe.copy()

        # ensure parameters are of correct type and length
        assert isinstance(columns, (pd.core.index.Index,
                                    list,
                                    np.array)),\
        "`columns` must be of type `pandas.core.index.Index` or `list`"

        assert isinstance(new_indices,
                          list),\
        "`reorder_indices` must be of type `list`"

        assert len(columns) == len(new_indices),\
        "Length of `columns` and `reorder_indices` must be equal"

        # check for negative values in `new_indices`
        if any(idx < 0 for idx in new_indices):

            # get a list of the negative values
            negatives = [value for value
                         in new_indices
                         if value < 0]

            # find the index location for each negative value in
            # `new_indices`
            negative_idx_locations = [new_indices.index(negative)
                                      for negative in negatives]

            # zip the lists
            negative_zipped = list(zip(negative_idx_locations,
                                       negatives))

            # replace the negatives in `new_indices` with their
            # absolute position in the index
            for idx, negative in negative_zipped:
                new_indices[idx] = df.columns.get_loc(df.columns[
                                                          negative])

        # re-order the index now
        # get all columns
        all_columns = df.columns

        # drop the columns that need to be re-indexed
        all_columns = all_columns.drop(columns)

        # now re-insert them at the specified locations
        zipped_columns = list(zip(new_indices,
                                  columns))

        for idx, column in zipped_columns:
            all_columns = all_columns.insert(idx,
                                             column)
        # re-index the dataframe
        df = df.ix[:, all_columns]

        print("Successfully re-indexed dataframe.")

    except Exception as e:
        print(e)
        print("Could not re-index columns. Something went wrong.")

    return df


def get_columns_by_dtype(dframe=None, skip_columns=None,
                         get_objects=True, get_integers=True,
                         get_floats=True, get_times=False,
                         combine_numeric=True):
    """
    Returns lists of different column dtypes of pandas dataframe
    according to the values passed to `column_types`.

    `dframe`: pandas dataframe

    `skip_columns`: list of columns or pandas.core.index.Index
    object; columns that should not be included in the returned lists

    `get_objects`: bool, specifies whether to gather string/object
    columns.

    `get_integers`: bool; specifies whether to gather integer
    columns; gets int32 and int64 columns.

    `get_floats`: bool; specifies whether to gather float columns;
    gets float32 and float64 columns.

    `get_times`: bool; specifies whether to gather time columns; not
    implemented currently

    `combine_numeric`: bool; specifies whether to combine integer and
    float columns into a single object; useful for further processing
    (e.g. casting all numeric to np.float32 at a later stage.)

    returns: `columns_list` which contains ones object for each
    boolean option specified upon calling the function.

    For example, if all booleans were true, the returned list would
    have 5 objects, one for each boolean condition.
    """
    # TODO Implement `get_times`
    print("Getting dataframe columns by dtype.")
    columns_list = []
    try:
        df = dframe.copy()
        # remove skip_columns (if any)
        if skip_columns:

            # drop skip_columns
            df = df.drop(skip_columns,
                         axis=1)

        if get_objects:
            # get object columns
            object_columns = df.iloc[:, :].\
                select_dtypes(include=['object']).columns
            columns_list.append(object_columns)

            print("Successfully gathered object columns.")

        if get_integers:
            # get int columns
            int_columns = df.iloc[:, :].\
                select_dtypes(include=['int32',
                                       'int64']).columns
            columns_list.append(int_columns)

            print("Successfully gathered integer columns.")

        if get_floats:
            # get float columns
            float_columns = df.iloc[:, :].\
                select_dtypes(include=['float32',
                                       'float64']).columns
            columns_list.append(float_columns)

            print("Successfully gathered float columns.")

        if get_times:
            raise NotImplementedError("`get_times` is not currently "
                                      "implemented.")

        if combine_numeric:
            # get float and int cols together
            non_object_columns = [column for column
                                  in df.iloc[:, :].columns
                                  if column not in object_columns]
            columns_list.append(non_object_columns)

            print("Successfully combined numeric columns together.")

    except Exception as e:
        print(e)
        print("Could not gather columns. Something went wrong.")

    return columns_list


def calculate_summary_stats(dframe=None, skip_columns=None):
    """
    Convenience function for calculating summary statistics for a
    pandas dataframe above and beyone what is normally returned by
    that dataframe method.

    Parameters
    ----------
    dframe: pandas dataframe; required.

    skip_columns: list or pandas.core.index.Index object; optional
        Columns to exclude when calculating summary statistics.

    # NOT IMPLEMENTED
    p_significance: float; required.

        Default: 0.05

        Significance level for tests like normality, etc. PASS/FAIL
        conditions are determined by comparing the test p-values to
        this significance level.

    Returns
    -------
    OrderedDict with each key corresponding to each column in the
    dataframe. Summary stats are nested dicts as well where the name
    of the statistic is the key and the statistic itself if the value.


    """
    # TODO: Input validation
    print("Calculating summary statistics.")

    # copy the dframe
    df = dframe.copy()

    # create the ordered dict
    summary_stats = OrderedDict()

    if skip_columns:
        df = df.drop(skip_columns,
                     axis=1)

    # get columns by dtype
    objs, ints, flts = get_columns_by_dtype(dframe=df,
                                            skip_columns=None,
                                            get_objects=True,
                                            get_integers=True,
                                            get_floats=True,
                                            combine_numeric=False)

    for column in df.columns:
        # add the column to the dict
        summary_stats[column] = OrderedDict(df.loc[:,
                                                 column].describe())

        # different feature types require different summary stats
        if column in objs:
            summary_stats[column]

        elif column in ints:
            # mask invalid values in order to calculate certain stats
            mask = np.ma.masked_invalid(df.loc[:, column])
            # attempt to calculate the stats
            summary_stats[column]['geometric mean'] = stats.gmean(mask)
            summary_stats[column]['mode'] = df.loc[:, column].mode()
            summary_stats[column]['median'] = df.loc[:, column].median()
            summary_stats[column]['iqr'] = \
                df.loc[:, column].quantile(0.75) - \
                df.loc[:, column].quantile(0.25)
            summary_stats[column]['skew'] = df.loc[:, column].skew()
            summary_stats[column]['kurtosis'] = \
                df.loc[:, column].kurtosis()
            summary_stats[column]['unique values'] = \
                df.loc[:, column].unique()
            summary_stats[column]['value counts'] = \
                df.loc[:, column].value_counts()
            summary_stats[column]['# missing'] = \
                df.loc[:, column].isnull().sum()
            summary_stats[column]['% missing'] = \
                df.loc[:, column].isnull().sum() /\
                df.loc[:, column].shape[0]

        elif column in flts:
            # mask invalid values in order to calculate certain stats
            mask = np.ma.masked_invalid(df.loc[:, column])
            # attempt to calculate the stats
            summary_stats[column]['geometric mean'] = stats.gmean(mask)
            summary_stats[column]['mode'] = df.loc[:, column].mode()
            summary_stats[column]['median'] = df.loc[:, column].median()
            summary_stats[column]['iqr'] = \
                df.loc[:, column].quantile(0.75) - \
                df.loc[:, column].quantile(0.25)
            summary_stats[column]['skew'] = df.loc[:, column].skew()
            summary_stats[column]['kurtosis'] = \
                df.loc[:, column].kurtosis()
            summary_stats[column]['# missing'] = \
                df.loc[:, column].isnull().sum()
            summary_stats[column]['% missing'] = \
                df.loc[:, column].isnull().sum() /\
                df.loc[:, column].shape[0]

    print("Successfully calculated summary stats.")

    return summary_stats


def fill_na(dframe=None, columns=None, fill_value=None):
    """
    Wrapper to replace missing values in a pandas dataframe.

    `dframe`: pandas dataframe

    `columns`: list of columns or pandas.core.index.Index object

    `fill_value`: str or int or float; specifies the value to fill
    missing cells with.

    """
    print("Filling missing values.")
    try:
        df = dframe.copy()
        df.loc[:, columns] = df.loc[:, columns].fillna(fill_value)
        print("Successfully filled missing values.")

    except Exception as e:
        print(e)
        print("Could not fill missing values. Something went wrong.")

    return df


def rounder(number=None):
    """
    Function to fix floating point rounding issues.
    """
    return int(number//1 + round((number%1)/0.5)//1)