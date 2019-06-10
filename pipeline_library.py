'''
Machine Learning Pipeline

Ben Fogarty

30 May 2019
'''

from copy import deepcopy
from textwrap import wrap
import json
import random
from dateutil.relativedelta import relativedelta
from sklearn import dummy, ensemble, linear_model, metrics, neighbors, svm, tree, preprocessing
import graphviz
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def read_csv(filepath, cols=None, col_types=None, index_col=None):
    '''
    Imports a CSV file into a pandas data frame, optionally specifying columns
    to import, the types of the columns, and an index column.

    Inputs:
    filepath (str): the path to the file
    cols (list of strings): an optional list of columns from the data to import;
        can only be used if the first line of the csv is a header row
    col_types (dict mapping strings to types): an optional dictionary specifying
        the data types for each column; each key must be a the name of a column
        in the dataset and the associated value must be a pandas datatype (valid
        types listed here: http://pandas.pydata.org/pandas-docs/stable/
        getting_started/basics.html#dtypes)
    index_col (str or list of strs): an optional column name or list of column
        names to index the rows of the dataframe with

    Returns: pandas dataframe
    '''
    return pd.read_csv(filepath, usecols=cols, dtype=col_types,
                       index_col=index_col)

def count_per_categorical(df, cat_column):
    '''
    Summaries the number of observations associated with each value in a given
    categorical column and shows the distribtuion of observations across
    categories.

    Inputs:
    df (pandas dataframe): the dataset
    cat_column (str): the name of the categorical column

    Returns: tuple of pandas dataframe, matplotlib figure
    '''
    df = df[~df[cat_column].isna()]
    count_per = df.groupby(cat_column)\
                  .count()\
                  .iloc[:, 0]\
                  .rename('obs_per_{}'.format(cat_column))

    summary = count_per.describe()
    fig = show_distribution(count_per)

    return summary, fig

def show_distribution(series):
    '''
    Graphs a histogram and the box plot of numeric type series and a bar plot
    of categorial type series.

    Inputs:
    df (pandas series): the variable to show the distribution of

    Returns: matplotlib figure

    Citations:
    Locating is_numeric_dtype: https://stackoverflow.com/questions/19900202/
    '''
    series = series.dropna()
    sns.set()
    if pd.api.types.is_numeric_dtype(series):
        f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        sns.distplot(series, kde=False, ax=ax1)
        sns.boxplot(x=series, ax=ax2, orient='h')
        ax1.set_title('Histogram')
        ax1.set_ylabel('Count')
        ax1.set_xlabel('')
        ax2.set_title('Box plot')
    else:
        f, ax = plt.subplots(1, 1)
        val_counts = series.value_counts()
        sns.barplot(x=val_counts.index, y=val_counts.values, ax=ax)
        ax.set_ylabel('Count')

    f.suptitle('Distribution of {}'.format(series.name))
    f.subplots_adjust(hspace=.5, wspace=.5)

    return f

def pw_correlate(df, variables=None, visualize=False):
    '''
    Calculates a table of pairwise correlations between numeric variables.

    Inputs:
    df (pandas dataframe): dataframe containing the variables to calculate
        pairwise correlation between
    variables (list of strs): optional list of variables to calculate pairwise
        correlations between; each passed str must be name of a numeric type
        (including booleans) column in the dataframe; default is all numeric
        type variables in the dataframe
    visualize (bool): optional parameter, if enabled the function generates
        a heat map to help draw attention to larger correlation coefficients

    Returns: pandas dataframe

    Wrapping long axis labels: https://stackoverflow.com/questions/15740682/
                               https://stackoverflow.com/questions/11244514/
    '''
    if not variables:
        variables = [col for col in df.columns
                     if pd.api.types.is_numeric_dtype(df[col])]

    corr_table = np.corrcoef(df[variables].dropna(), rowvar=False)
    corr_table = pd.DataFrame(corr_table, index=variables, columns=variables)

    if visualize:
        sns.set()
        f, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_table, annot=True, annot_kws={"size": 'small'},
                    fmt='.2f', linewidths=0.5, vmin=-1, vmax=1, square=True,
                    cmap='coolwarm', ax=ax)

        labels = ['-\n'.join(wrap(l.get_text(), 16)) for l in ax.get_yticklabels()]
        ax.set_yticklabels(labels)
        labels = ['-\n'.join(wrap(l.get_text(), 16)) for l in ax.get_xticklabels()]
        ax.set_xticklabels(labels)
        ax.tick_params(axis='both', rotation=0, labelsize='small')
        ax.tick_params(axis='x', rotation=90, labelsize='small')

        ax.set_title('Correlation Table')
        f.tight_layout()
        f.show()

    return corr_table

def summarize_data(df, grouping_vars=None, agg_cols=None):
    '''
    Groups rows based on the set of grouping variables and report summary
    statistics over the other numeric variables.

    Inputs:
    df (pandas dataframe): dataframe containing the variables to calculate
        pairwise correlation between
    grouping_vars (list of strs): optional list of variables
        to group on before aggregating; each passed str must be name of a column
        in the dataframe; if not included, no grouping is performed
    agg_cols (list of strs): optional list of variables to
        aggregate after grouping; each passed str must be name of a column in
        the dataframe; default is all numeric type variables in the dataframe

    Returns: pandas dataframe
    '''
    if not grouping_vars:
        grouping_vars = []
    if agg_cols:
        keep = grouping_vars + agg_cols
        df = df[keep]

    if grouping_vars:
        summary = df.groupby(grouping_vars)\
                    .describe()
    else:
        summary = df.describe()

    return summary.transpose()

def find_ouliers_univariate(series):
    '''
    Identifies values in a series that fall more than 1.5 * IQR below the first
    quartile or 1.5 * IQR above the third quartile.

    Inputs:
    series (pandas series): the series to look for outliers in, must be numeric

    Returns: pandas series
    '''
    quartiles = np.quantile(series.dropna(), [0.25, 0.75])
    iqr = quartiles[1] - quartiles[0]
    lower_bound = quartiles[0] - 1.5 * iqr
    upper_bound = quartiles[1] + 1.5 * iqr

    return (lower_bound > series) | (upper_bound < series)

def find_outliers(df, excluded=None):
    '''
    Identifies outliers for each numeric column in a dataframe, and returns a
    dataframe matching each record with the columns for which it is an outlier
    and the number and percent of checked columns for which a is an outlier.
    Outlier is defined as any value thats fall more than 1.5 * IQR below the
    first quartile or 1.5 * IQR above the third quartile of all the values in a
    column.

    Inputs:
    df (pandas dataframe): the dataframe to find outliers in
    excluded (str list of strs): optional column name or a list of columns names
        not to look for outliers in; default is including all numeric columns

    Returns: pandas series
    '''
    if not excluded:
        excluded = []

    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)

    outliers = df[numeric_cols]\
                 .drop(excluded, axis=1, errors='ignore')\
                 .apply(find_ouliers_univariate, axis=0)
    outliers['Count Outlier'] = outliers.sum(axis=1, numeric_only=True)
    outliers['% Outlier'] = (outliers['Count Outlier'] /
                             (len(outliers.columns) - 1) * 100)

    return outliers

def identify_missing(series):
    '''
    Generates series specifying whether observation contains a missing value.

    Inputs:
    sereis (pandas dataframe): the variable to identify missing in

    Returns: pandas series
    '''
    return series.isnull()

def impute_missing(series, method=None, manual_val=None):
    '''
    Replaces missing values in a series with using the specified imputation
    method. Currently supported imputation mentions include mean, median, mode,
    and manual. If the imputation method is not specified for a given column,
    median is used for numeric columns and mode is used for non-numeric columns.

    Inputs:
    series (pandas series): the series to impute missing data for
    method (str): the imputation method; currently supported methods:
        - 'mean': fill missing with the mean of the column (numeric series only)
        - 'median': fill missing with the median of the column (numeric series
                    only)
        - 'mode': fill missing with the mode of the column
        - 'manual': fill missing with a user-specified value
    manual_val (single value, should match type of series): a user-specified
        value to fill missing values with

    Returns: pandas series
    '''
    if manual_val is None:
        manual_val = 'N/A'

    if method == 'mean':
        mean = np.mean(series)
        return series.fillna(mean)
    elif method == 'median':
        median = np.median(series.dropna())
        return series.fillna(median)
    elif method == 'mode':
        mode = series.mode().iloc[0]
        return series.fillna(mode)
    elif method == 'manual':
        return series.fillna(manual_val)

def preprocess_data(df, methods=None, manual_vals=None):
    '''
    Removes missing values and adds columns to the dataframe to identify which
    observations where missing certain variables. If the imputation method for
    a column is not specified, median is used for numeric variables and mode is
    used for non-numeric variables.

    Inputs:
    df (pandas dataframe): contains the data to preprocess
    methods (dict): keys are column names and values the imputation method to
        apply to that column; currently supported methods:
        - 'mean': fill missing with the mean of the column (numeric series only)
        - 'median': fill missing with the median of the column (numeric series
                    only)
        - 'mode': fill missing with the mode of the column
        - 'manual': fill missing with a user-specified value
    manual_vals (dict): keys are column names and values the values to fill
        missing values with in columns with 'manual' imputation method

    Returns: pandas dataframe
    '''
    if methods is None:
        methods = {}
    if manual_vals is None:
        manual_vals = {}

    to_process = list(methods.keys())
    missing = df[to_process]\
                .apply(identify_missing)\
                .add_suffix('_missing')
    processed_cols = df[to_process]\
                       .apply(lambda x: impute_missing(x, method=methods.get(x.name, None),
                              manual_val=manual_vals.get(x.name, None)),
                              axis=0)
    df = df.drop(to_process, axis=1)
    return pd.concat([df, processed_cols, missing], axis=1)

def cut_binary(series, threshold, or_equal_to=False):
    '''
    Cuts a continuous variable into a binary variable based on whether or not
    each observation is above (True) or below (False) some threshold.

    series (pandas series): the variable to cut
    threshold (numeric type): observations are marked as true if above this
        threshold and false if below this threshold
    or_equal_to (bool): if false, observations equal to the threshold are marked
        as fasle; if true, observations equal to the threshold are marked as
        true

    Returns: pandas series
    '''
    if not pd.api.types.is_numeric_dtype(series):
        series = series.astype(float)

    if or_equal_to:
        return  series >= threshold
    else:
        return series > threshold

def cut_variable(series, bins, labels=None, kwargs=None):
    '''
    Discretizes a continuous variable into bins. Bins are half-closed, [a, b).

    Inputs:
    series (pandas series): the variable to discretize
    bins (int or list of numerics): the binning rule:
        - if int: cuts the variable into approximate n bins with an approximately
          equal number of observations (there may be fewer bins or some bins
          with a substantially larger number of observations depending on the
          distribution of the data)
        - if list of numerics: cuts the variable into bins with edges determined
          by the sorted numerics in the list; any values not covered by the
          specified will be labeled as missing
    labels (list of str): optional list of labels for the bins; must be the same
        length as the number of bins specified
    kwargs (dictionary): keyword arguments to pass to either pd.cut or pd.qcut

    Return: tuple of pandas series and numpy array of bin edges
    '''
    if not kwargs:
        kwargs = {}

    if isinstance(bins, int):
        return pd.qcut(series, bins, labels=labels, duplicates='drop',
                       retbins=True, **kwargs)\
                 .astype('category')

    return pd.cut(series, bins, labels=labels, include_lowest=True, **kwargs)\
             .astype('category')

def create_dummies(df, column, values=None):
    '''
    Transforms variables into a set of dummy variables.

    Inputs:
    df (pandas dataframe/series): the data to transform dummies in;
        all columns not being converted to dummies must be numeric
        types
    columns (list of strs): column name containing categorical
        variable to convert to dummy variables
    values (list of values): values in the specified column to create dummies
        for; by default, a column is made for all values

    Returns: pandas dataframe where the columns to be converted is replaced with
        columns containing dummy variables
    '''
    if values is None:
        values = list(df[column].value_counts().index)
    for value in values:
        df[column + '_' + value] = df[column] == value

    df = df.drop(column, axis=1)

    return df

def scale_variable_minmax(series, a=None, b=None):
    '''
    Scales a variable according to the formula (2x - (a + b))/(a - b),
    where x is an observation from the dataset.

    If a is not specified, then the maximum value of the variable is a, and if b
    is not specified, then the minimum value of the variable is b.

    Inputs:
    series (pandas series): the variable to scale; must be a numeric type series
    a (numeric type): manual parameter for the scaling formula
    b (numeric type): manual parameter for the scaling formula
    '''
    if a is None:
        a = max(series)
    if b is None:
        b = min(series)

    if (a - b) == 0:
        return 0

    return (2 * series - (a + b)) / (a - b)

def generate_n_occurences(series, addl_obs=None):
    '''
    Generates a new series where each instance is linked to the number of
    observations that have the same value as that instance in the original
    series.

    Inputs:
    series (pandas series): the original variable to generate the feature based
        on
    addl_obs (pandas series): additional observations to consider when counting the
        number of occurences of each value; these observations are not included
        in the output, and this column is indended to be used for observations
        from the training set when the series is from a test set
    '''
    val_counts = series.append(addl_obs, ignore_index=True)\
                       .value_counts()

    return series.map(val_counts)

def create_time_diff(start_dates, end_dates):
    '''
    Calculates the time difference between two date columns.

    Inputs:
    start_dates (pandas series): the start dates to calculate the difference
        from; column should be have type datetime
    end_dates (pandas series): the end dates to calculate the difference to;
        columns should have type datetime

    Returns: pandas series of timedelta objects
    '''
    return end_dates - start_dates

def report_n_missing(df):
    '''
    Reports the percent of missing (float.NaN or None) values for each column
    in a dataframe.

    Inputs:
    df (pandas dataframe): the dataset

    Returns: pandas dataframe
    '''
    missing = pd.DataFrame(columns=['# Missing'])

    for column in df.columns:
        missing.loc[column, '# Missing'] = np.sum(df[column].isna())

    missing['% Missing'] = missing['# Missing'] / len(df) * 100

    return missing

def visualize_decision_tree(dt, feature_names, class_names, filepath='tree'):
    '''
    Saves and opens a PDF visualizing the specified decision tree.

    Inputs:
    dt (sklearn.tree.DecisionTreeClassifier): a trained decision tree classifier
    feature_names (list of strs): a list of the features the data was trained
        with; must be in the same order as the features in the dataset
    class_names (list of strs): a list of the classes of the target attribute
        the model is predicting; must match the target attribute values if
        those values were given in ascending order
    filepath (str): optitional parameter specifying the output path for the
        visualization (do not include the file extension); default is 'tree' in
        the present working directory

    Citations:
    Guide to sklearn decision trees: https://scikit-learn.org/stable/modules/
        tree.html
    sklearn.tree.export_graphviz docs: https://scikit-learn.org/stable/modules/
        generated/sklearn.tree.export_graphviz.htm
    '''
    dot_data = tree.export_graphviz(dt, None, feature_names=feature_names,
                                    class_names=class_names, filled=True)
    graph = graphviz.Source(dot_data)
    output_path = graph.render(filename=filepath, view=True)

def generate_iter_model_specs(base_specs, iter_param, iter_vals):
    '''
    Produces a list of model specification dictionaries for use with
    generate_classifer) that iterates over different values for one parameter.

    Inputs:
    base_specs (dict): a dictionary specifying the base parameters of the
        model that are not iterated over. Each dictionary must contain a "model"
        key with a value specifying the type of model to generate; currently
        supported types are listed below. All other entries in the dictionary
        are optional and should have the key as the name of a parameter for the
        specified classifier and the value as the desired value of that
        parameter.
    iter_param (str): the name of the parameter to iterate over
    iter_vals (list): the values of the iterative parameter to generate model
        specifications with

    Currently supported model types:
    'dt': sklearn.tree.DecisionTreeClassifier
    'lr': sklearn.linear_model.LogisticRegression
    'knn': sklearn.neighbors.KNeighborsClassifier
    'svc': sklearn.svm.LinearSVC
    'rf': sklearn.ensemble.RandomForestClassifier
    'boosting': sklearn.ensemble.AdaBoostClassifier
    'bagging': sklearn.ensemble.BaggingClassifier
    'dummy': sklearn.dummy.DummyClassifier

    Example usage:
    generate_classifiers(x, y, {'model': 'dt', 'max_depth': 5})

    The above line will generate a decision tree classifiers with a max depth of
    5.

    For more information on valid parameters to include in the dictionaries,
    consult the sklearn documentation for each model.
    '''
    models = []
    for val in iter_vals:
        model_specs = deepcopy(base_specs)
        model_specs[iter_param] = val
        models.append(model_specs)
    return models

def write_model_specs(models, output_path, input_path=None):
    '''
    Writes a list of models to file so that they can be reused later. As the
    output is stored in JSON format, model specifications including callable
    objects are currently not supported.

    Inputs:
    models (list of dicts): the model specifications to write to file
    output_path (str): the filepath to output the model specifications to, will
        overwrite any existing file
    input_path (str): a file with existing model specs to append the list to
    '''
    if input_path is not None:
        with open(input_path, 'r') as input_file:
            existing_models = json.load(input_file)
            models = existing_models + models

    with open(output_path, 'w') as output_file:
        json.dump(models, output_file)

def generate_classifier(features, target, model_specs):
    '''
    Generates a classifier to predict a target attribute (target)
    based on other attributes (features).

    Inputs:
    features (pandas dataframe): Data for features to build the classifier(s)
        with; all columns must be numeric in type
    target (pandas series): Data for target attribute to build the classifier(s)
        with; should be categorical data in a numerical form
    model_specs (dicts): A dictionary specifying the classifier
        model to generate. Each dictionary must contain a "model" key with a
        value specifying the type of model to generate; currently supported
        types are listed below. Optionally, a "name" key can be specified to
        help identify the model, however, this key will not be used in this
        function. All other entries in the dictionary are optional
        and should have the key as the name of a parameter for the specified
        classifier and the value as the desired value of that parameter.

    Returns: tuple trained classifier objects and list of the integer indices
        of the features used to train the model

    Currently supported model types:
    'dt': sklearn.tree.DecisionTreeClassifier
    'lr': sklearn.linear_model.LogisticRegression
    'knn': sklearn.neighbors.KNeighborsClassifier
    'svc': sklearn.svm.LinearSVC
    'rf': sklearn.ensemble.RandomForestClassifier
    'boosting': sklearn.ensemble.AdaBoostClassifier
    'bagging': sklearn.ensemble.BaggingClassifier
    'dummy': sklearn.dummy.DummyClassifier

    Example usage:
    generate_classifiers(x, y, {'model': 'dt', 'max_depth': 5})

    The above line will generate a decision tree classifiers with a max depth of
    5.

    For more information on valid parameters to include in the dictionaries,
    consult the sklearn documentation for each model.
    '''
    model_class = {'dt': tree.DecisionTreeClassifier,
                   'lr': linear_model.LogisticRegression,
                   'knn': neighbors.KNeighborsClassifier,
                   'svc': svm.LinearSVC,
                   'rf': ensemble.RandomForestClassifier,
                   'boosting': ensemble.GradientBoostingClassifier,
                   'bagging': ensemble.BaggingClassifier,
                   'dummy': dummy.DummyClassifier}

    model_type = model_specs['model']
    model_specs = {key: val for key, val in model_specs.items()\
                   if not key in ['model', 'name']}
    model = model_class[model_type](**model_specs)
    model.fit(features, target)

    return model

def predict_target_probability(model, features):
    '''
    Generates predicted probabilities of a binary target being positive
    (represented as 1) based on a model.

    model (trained sklearn classifier): the model to generate predicted
        probabilities with
    features (pandas dataframe): instances to generate predictied probabilities
        for; structure of the data (columns and column types) must match the
        data used to train the model

    Returns: pandas series
    '''
    if isinstance(model, svm.LinearSVC):
        pred_probs = model.decision_function(features)
    else:
        pred_probs = model.predict_proba(features)[:, 1]

    return pd.Series(data=pred_probs, index=features.index)

def predict_target_class(pred_probs, threshold, tie_breaker='random',
                         true_classes=None, seed=None):
    '''
    Generates predicted probabilities of a binary target being positive
    (represented as 1) based on a model.

    pred_probs (pandas series): predicted probabilies of the target variable
        being positive
    threshold (float): the precentile of observations to predict as positive,
        should be in the range [0.0, 1.0]
    tie_breaker (str): how to break ties when predicting classes at the margin
        when predicting classese; valid inputs are:
        - 'random': randomly selects which instances to predict as positive among
                    those with the lowest probability meeting the specified
                    threshold
        - 'pessimistic': prioritizes selecting which instances with a true
                         target value of negative to predict as positive among
                         those with the lowest probability meeting the specified
                         threshold, used for evaluation
        - 'optimistic': prioritizes selecting instances with a true
                         target value of positive to predict as positive among
                         those with the lowest probability meeting the specified
                         threshold, used for evaluation
    true_classes (pandas series): the ground truth about whether the target
        variable is positive
    seed (int): optional seed to make results reproducable

    Returns: pandas series
    '''
    assert tie_breaker in ['random', 'pessimistic', 'optimistic']
    if tie_breaker != 'random':
        assert true_classes is not None

    max_positives = int(np.floor(threshold * len(pred_probs)))
    pred_classes = pred_probs >= np.quantile(pred_probs, 1 - threshold,
                                             interpolation='higher')
    n_positives = np.sum(pred_classes)
    excess_positives = n_positives - max_positives
    if not excess_positives:
        return pred_classes

    min_positive_prob = min(pred_probs[pred_classes])
    if tie_breaker == 'random':
        change_pred = pred_probs[pred_probs == min_positive_prob]\
                                .sample(n=excess_positives, random_state=seed)\
                                .index
        pred_classes.loc[change_pred] = False
        return pred_classes

    if tie_breaker == 'pessimistic':
        prioritize = ~true_classes.astype(bool)
    elif tie_breaker == 'optimistic':
        prioritize = true_classes.astype(bool)
    change_pred = pred_probs[pred_probs == min_positive_prob]\
                            [~prioritize]\
                            .index
    pred_classes.loc[change_pred] = False
    n_positives = np.sum(pred_classes)
    excess_positives = n_positives - max_positives

    if not excess_positives:
        return pred_classes
    if excess_positives < 0:
        change_pred = pred_probs[pred_probs == min_positive_prob]\
                                [~prioritize]\
                                .sample(n=-excess_positives, random_state=seed)\
                                .index
        pred_classes.loc[change_pred] = True
    elif excess_positives > 0:
        change_pred = pred_probs[pred_probs == min_positive_prob]\
                                [prioritize]\
                                .sample(n=excess_positives, random_state=seed)\
                                .index
        pred_classes.loc[change_pred] = False

    return pred_classes

def evaluate_classifier(pred_probs, true_classes, thresholds, tie_breaker='random',
                        seed=None, model_name=None, dataset_name=None):
    '''
    Calculates a number of evaluation metrics (accuracy, precision, recall, and
    F1 at different levels and AUC-ROC) and generates a graph of the
    precision-recall curve for a given model.

    pred_probs (pandas series): predicted probabilies of the target variable
        being positive
    true_classes (pandas series): the ground truth about whether the target variable
        is positive
    thresholds (list of floats): different threshold levels to use when
        calculating precision, recall and F1, should be in range [0.0, 1.0]
    tie_breaker (str): how to break ties when predicting classes at the margin
        when predicting classese; valid inputs are:
        - 'random': random selects which instances to predict as positive among
                    those with the lowest probability meeting the specified
                    threshold
        - 'pessimistic': prioritizing selecting which instances with a true
                         target value of negative to predict as positive among
                         those with the lowest probability meeting the specified
                         threshold
        - 'optimistic': prioritizing selecting instances with a true
                         target value of positive to predict as positive among
                         those with the lowest probability meeting the specified
                         threshold
    seed (int): optional seed to make results reproducable
    model_name (str): optional model name to include in the title of the
        precision/recall curve graph
    dataset_name (str): optional model name to include in the title of the
        precision/recall curve graph

    Returns: tuple of pandas series and matplotlib figure
    '''
    index = [['Accuracy'] * len(thresholds) +['Precision'] * len(thresholds) +
             ['Recall'] * len(thresholds) + ['F1'] * len(thresholds),
             thresholds * 4]
    index = list(zip(*index))
    index.append(('AUC-ROC', None))
    index = pd.MultiIndex.from_tuples(index, names=['Metric', 'Threshold'])
    evaluations = pd.Series(index=index)

    for threshold in thresholds:
        pred_classes = predict_target_class(pred_probs, threshold, tie_breaker,
                                            true_classes, seed)
        evaluations['Accuracy', threshold] = metrics.accuracy_score(true_classes, pred_classes)
        evaluations['Precision', threshold] = metrics.precision_score(true_classes, pred_classes)
        evaluations['Recall', threshold] = metrics.recall_score(true_classes, pred_classes)
        evaluations['F1', threshold] = metrics.f1_score(true_classes, pred_classes)

    evaluations['AUC-ROC', None] = metrics.roc_auc_score(true_classes, pred_classes)

    fig = graph_precision_recall(pred_probs, true_classes, tie_breaker=tie_breaker,
                                 seed=seed, model_name=model_name,
                                 dataset_name=dataset_name)

    return evaluations, fig

def graph_precision_recall(pred_probs, true_classes, resolution=33,
                           tie_breaker='random', seed=None, model_name=None,
                           dataset_name=None):
    '''
    Produces a precision/recall graph based on predictions generated by a model.

    pred_probs (pandas series): predicted probabilies of the target variable
        being positive
    true_classes (pandas series): the ground truth about whether the target variable
        is positive
    resolution (list of ints): number of evenly-spaced threshold levels to plot
        recall and precision at
    tie_breaker (str): how to break ties when predicting classes at the margin
        when predicting classese; valid inputs are:
        - 'random': random selects which instances to predict as positive among
                    those with the lowest probability meeting the specified
                    threshold
        - 'pessimistic': prioritizing selecting which instances with a true
                         target value of negative to predict as positive among
                         those with the lowest probability meeting the specified
                         threshold
        - 'optimistic': prioritizing selecting instances with a true
                         target value of positive to predict as positive among
                         those with the lowest probability meeting the specified
                         threshold
    seed (int): optional seed to set for use with random tiebreaking
    model_name (str): optional model name to include in the title of the
        precision/recall curve graph
    dataset_name (str): optional model name to include in the title of the
        precision/recall curve graph

    Returns: tuple of pandas series and matplotlib figure
    '''
    if not seed:
        seed = random.randrange(0, 2147483647) #must set some seed for
        #graph to make sense, given repeated calls to predict_target_class
    sns.set()
    fig, ax = plt.subplots()
    thresholds = np.linspace(0.01, 1, num=resolution)
    precision = []
    recall = []
    for threshold in thresholds:
        pred_classes = predict_target_class(pred_probs, threshold, tie_breaker,
                                            true_classes, seed)
        precision.append(metrics.precision_score(true_classes, pred_classes))
        recall.append(metrics.recall_score(true_classes, pred_classes))
    precision_recall_curves = pd.DataFrame
    sns.lineplot(thresholds, precision, drawstyle='steps-pre', ax=ax, label='Precision')
    sns.lineplot(thresholds, recall, drawstyle='steps-pre', ax=ax, label='Recall')

    ax.set_xlabel('Threshold')
    ax.set_ylabel('Precision')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.01])

    if model_name and dataset_name:
        fig.suptitle('Precision-Recall Curves: {}, {}'.format(model_name, dataset_name))
    elif model_name:
        fig.suptitle('Precision-Recall Curves: {}'.format(model_name))
    elif dataset_name:
        fig.suptitle('Precision-Recall Curves: {}'.format(dataset_name))
    else:
        fig.suptitle('Precision-Recall Curves')

    return fig

def create_temporal_splits(df, date_col, time_length, gap=None, start_date=None,
                           end_date=None):
    '''
    Splits into different sets by time intervals.
    Inputs:
    df (pandas dataframe): the full dataset to split
    date_col (str): the name of the column in the dataframe containing the date
        attribute to split on
    time_length (dictionary): specifies the time length of each split, with
        strings of units of time (i.e. hours, days, months, years, etc.) as keys
        and integers as values; for example 6 months would be {'months': 6}
    gap (dictionary): optional length of time to leave between the end of the
        training set and the beginning of the test set, specified as a dictionary
        with string units of time as keys and integers as values
    start_date (str): the first date to include in a testing split; value should
        be in the form "yyyy-mm-dd", if blank the first date in a training set
        will be the first date in the data set plus the value of time_length
    end_date (str): the final date to include in a testing split; value should
        be in the form "yyyy-mm-dd", if blank the first date in a training set
        will be the first date in the data set plus the value of time_length
    Returns: tuple of list of pandas dataframes, the first of which contains
        test sets and the second of which contains training sets
    '''
    time_length = relativedelta(**time_length)

    if gap:
        gap = relativedelta(**gap)
    else:
        gap = relativedelta()
    if start_date:
        start_date = pd.to_datetime(start_date)
    else:
        start_date = min(df[date_col]) + time_length
    if end_date:
        end_date = pd.to_datetime(end_date)
    else:
        end_date = max(df[date_col].apply(lambda x: x - gap))


    test_splits = []
    train_splits = []
    i = 0
    while start_date + (i * time_length) <= end_date:
        test_date = start_date + (i * time_length)
        test_mask = df[date_col] == test_date
        train_mask = df[date_col] <= (test_date - gap)
        test_splits.append(df[test_mask])
        train_splits.append(df[train_mask])
        i += 1

    return train_splits, test_splits

def get_feature_importance(X_train, clf, model):
    '''
    clf: a classfier object
    model_type: model type abbreviation

    'dt': sklearn.tree.DecisionTreeClassifier
    'lr': sklearn.linear_model.LogisticRegression
    'knn': sklearn.neighbors.KNeighborsClassifier
    'svc': sklearn.svm.LinearSVC
    'rf': sklearn.ensemble.RandomForestClassifier
    'boosting': sklearn.ensemble.AdaBoostClassifier
    'bagging': sklearn.ensemble.BaggingClassifier
    'dummy': sklearn.dummy.DummyClassifier

    '''
    model_type = model['model']
    if model_type in ['rf', 'dt', 'boosting', 'bagging']:
        importances = list(zip(X_train.columns, clf.feature_importances_))
    elif model_type in ['lr', 'svm']:
        importances = list(zip(X_train.columns, clf.coef_[0]))
    else:
        importances = None

    rv = pd.DataFrame(importances)
    rv.columns = ['Feature', 'Importance']
    rv = rv.set_index('Feature')
    return rv

def create_interactions(df, cols_to_interact):
    '''
    Create feature interactions
    Build case /default change type
    Specify interaction
    '''
    interaction = 'inter_' + ''.join(cols_to_interact)
    df[interaction] = df[cols_to_interact[0]].astype(int)
    for col in cols_to_interact[1:]:
        if isinstance(col, int):
            df[interaction] = df[interaction] * col

    return df

def days_between(date, ref_date):
    '''
    Difference in days with respect to a reference period; here: time buckets
    Build case /default change type
    Ensure that they are 
    '''
    date = pd.to_datetime(date)
    ref_date = pd.to_datetime(ref_date)
    duration = (ref_date - date).dt.days

    return duration

def convert_iter_dummy(df, col):
    '''
    Converts a columns in a data frame with iterables (list, set, etc.) to a 
    set of dummy columns.

    Inputs:
    df (pandas dataframe): the dataset
    col (column name): the name of the column containing the iterables

    Returns: pandas dataframe
    '''
    mlb = preprocessing.MultiLabelBinarizer()

    rv = pd.DataFrame(mlb.fit_transform(df[col]),
                      columns=list(map(lambda x: col + '_' + x, mlb.classes_)),
                      index=df.index)
    df = df.drop(col, axis=1)

    return pd.concat([df, rv], axis=1)