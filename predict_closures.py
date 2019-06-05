'''
Predicting Donors Choose Funding

Ben Fogarty

2 May 2019
'''

import argparse
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import license_clean
import pipeline_library as pl




def apply_pipeline(preprocessing, features, models, dataset=None, seed=None,
                   save_figs=False):
    '''
    Applies the pipeline library to predicting if a project on Donors Choose
    will not get full funding within 60 days.

    Inputs:
    dataset (str): path to the pickle file containing the training data
    preprocessing (dict): dictionary of keyword arguments to pass to the
        preprocess_data function
    seed (str): seed used for random process to adjucate ties when translating
        predicted probabilities to predicted classes given some percentile
        threshold
    save_figs (bool): if true figures are saved instead of displayed
    '''
    ## Optional overide if dataset is specified?
    if dataset is None:
        df = license_clean.get_lcs_data() #parameterize for median homevalue/cta?, pass buckets?
    else:
        df = pickle.load(open(dataset, "rb" ))

    print('Generating training/testing splits...')
    training_splits, testing_splits = pl.create_temporal_splits(data=df, time_period_col='time_period')


    print('Preprocessing data and generating features...')
    for i in range(len(training_splits)):
        training_splits[i] = preprocess_data(training_splits[i], **preprocessing)
        testing_splits[i] = preprocess_data(testing_splits[i], **preprocessing)

        training_splits[i], testing_splits[i] = generate_features(training_splits[i],
                                                                  testing_splits[i],
                                                                  **features)
    for i in range(len(models)):
        model = models[i]
        print('-' * 20 +  '\nModel Specifications\n' + str(model) + '\n' + '_' * 20)
        model_name = model.get('name', 'Model #{}'.format(i + 1))
        trained_classifiers = train_classifiers(model, training_splits)
        pred_probs = predict_probs(trained_classifiers, testing_splits)
        if save_figs:
            evaluate_classifiers(pred_probs, testing_splits, seed, model_name,
                                 fig_prefix=model_name)
        else:
            evaluate_classifiers(pred_probs, testing_splits, seed, model_name)

def transform_data(df):
    '''
    Changes the types of columns in the dataset and creates new columns to
    allow for better data exploration and modeling.

    Inputs:
    df (pandas dataframe): the dataset

    Returns: pandas dataframe
    '''

    return df

def preprocess_data(df, methods=None, manual_vals=None):
    '''
    Preprocesses the data

    Inputs:
    df (pandas dataframe): the dataset
    methods (dict): keys are column names and values the imputation method to
        apply to that column; valid methods are defined in pipeline_library
    manual_vals (dict): keys are column names and values the values to fill
        missing values with in columns with 'manual' imputation method

    Returns: pandas dataframe
    '''
    df = pl.preprocess_data(df, methods=methods, manual_vals=manual_vals)

    return df

def generate_features(training, testing, n_ocurr_cols, scale_cols, bin_cols,
                      dummy_cols, iter_dummy_cols, binary_cut_cols, 
                      duration_cols, interaction_cols, drop_cols):
    '''
    Generates categorical, binary, and scaled features. While features are
    generate for the training data independent of the testing data, features
    for the testing data sometimes require ranges or other information about the
    properties of features created for the training data to ensure consistency.
    Operations will occurr in the following order:
    - Create number of occurences columns, name of each column will be the name
      of the original column plus the suffix '_n_ocurr' (new column created)
    - Scale columns (new column with name of original column + '_scale')
    - Bin columns (replaces original column)
    - Create dummy columns (new column with name of original + '_tf')
    - Create dummy columns for iterables (replaces original column)
    - Binary cut columns (replaces original column)
    - Duration columns (new column with original name + '_duration')
    - Create interaction columns ()
    - Drop columns (eliminates original column)

    As such, number of occurence columns may be scaled, binned, etc, by
    specifying '<col_name>_n_ocurr' in the arguments. Binned columns will
    automatically be converted to dummies

    Inputs:
    training (pandas dataframe): the training data
    testing (pandas dataframe): the testing data
    n_ocurr_cols (list of strs): names of columns to count the number of
        ocurrences of each value for
    scale_cols (list of strs): names of columns to rescale to be between -1 and
        1
    bin_cols (dict): each key is the name of a column to bin and each value is a
        dictionary of arguments to pass to the cut_variable function in
        pipeline_library (must contain a value for bin (a binning rule),
        labels and kwargs parameters are optional)
    dummy_cols (list of strs): names of columns to convert to dummy variables
    iter_dummy_cols (list of col names): name of columns where each value is an
        iterable to be converted to a set of dummy columns
    duration_cols (list of tuples of column names): first column is name of column containg
        start date, second column is name of column containing end dates
    interaction_cols (list of n-ples of col names): each tuple contains names of
        columns to interact with one another
    binary_cut_cols (dict of dicts): each key is the name of a column to cut
        into two groups based on some threshold and each value is a dictionry
        of arguments to pass to the cut_binary function in pipeline_library
        (must contain a value for threshold, or_equal_to parameter is optional)
    drop_cols (list of strs): names of columns to drop

    Returns: tuple of pandas dataframe, the training and testing datasets after
        generating the features
    '''
    '''
    df = df.drop(['school_longitude', 'school_latitude', 'schoolid',
                  'teacher_acctid', 'school_district', 'school_ncesid',
                  'school_county'],
                  axis=1)
    '''
    for col in n_ocurr_cols:
        training.loc[:, str(col) + '_n_occur'] = pl.generate_n_occurences(training[col])
        testing.loc[:, str(col) + '_n_occur'] = pl.generate_n_occurences(testing[col],
                                                             addl_obs=training[col])

    for col in scale_cols:
        max_training = max(training[col])
        min_training = min(training[col])
        training.loc[:, col + '_scale'] = pl.scale_variable_minmax(training[col], a=max_training,
                                                                   b=min_training)
        testing.loc[:, col + '_scale'] = pl.scale_variable_minmax(testing[col], a=max_training,
                                                       b=min_training)

    for col, specs in bin_cols.items():
        training.loc[:, col], bin_edges = pl.cut_variable(training[col], **specs)
        bin_edges[0] = - float('inf') #test observations below the lowest observation
        #in the training set should be mapped to the lowest bin
        bin_edges[-1] = float('inf') #test observations above the highest observation
        #in the training set should be mapped to the highest bin
        testing[col], _ = pl.cut_variable(testing[col], bin_edges)

    dummy_cols += list(bin_cols.keys())
    for col in dummy_cols:
        values = list(training[col].value_counts().index)
        training = pl.create_dummies(training, col, values=values)
        testing = pl.create_dummies(testing, col, values=values)

    for col in iter_dummy_cols:
        training = pl.convert_iter_dummy(training, col)
        training_cols = set(training.columns)
        testing = pl.convert_iter_dummy(testing, col)
        testing_cols = set(testing.columns)
        extra_testing_cols = testing_cols - training_cols
        testing = testing.drop(extra_testing_cols, axis=1)
        missing_testing_cols = training_cols - testing_cols
        for missing_col in missing_testing_cols:
            testing[missing_col] = 0



    for start_col, end_col in duration_cols:
        training[start_col + '-' + end_col + "_duration"] = pl.days_between(training[start_col], 
                                                                          training[end_col])
        testing[start_col + '-' + end_col + "_duration"] = pl.days_between(testing[start_col],
                                                                         testing[end_col])

    for col, specs in binary_cut_cols.items():
        training[col + '_tf'] = pl.cut_binary(training[col], **specs)
        testing[col + '_tf'] = pl.cut_binary(testing[col], **specs)

    for cols in interaction_cols:
        testing = pl.create_interactions(training, cols)
        training = pl.create_interactions(testing, cols)

    training = training.drop(drop_cols, axis=1)
    testing = testing.drop(drop_cols, axis=1)

    return training, testing

def train_classifiers(model, training):
    '''
    Returns a 2-D list that where where each inner list is a set of
    classifiers and the outer list represents each training/test set (i.e.
    at location 0,0 in the output list is the first model trained on the
    first set and at location 1,0 is the first model trained on the second
    set).

    Inputs:
    models (dict): specifications for the classifiers
    training (list of pandas dataframe): a list of training datasets

    Returns: 2D list of trained sklearn classifiers
    '''
    classifiers = []
    for i in range(len(training)):
        print('Building with training set {}'.format(i + 1))
        features = training[i].drop('no_renew_nextpd', axis=1)
        target = training[i].no_renew_nextpd
        classifiers.append(pl.generate_classifier(features, target, model))

    return classifiers

def predict_probs(trained_classifiers, testing_splits):
    '''
    Generates predictions for the observations in the i-th training split based
    on the i-th trained classifier.

    Inputs:
    trained_classifiers (list of sklearn classifers): the i-th model should have
        been trained on the i-th sklearn training split
    testing_splits (list of pandas dataframe): the i-th testing split should be
        associated with the i-th training split

    Returns: list of pandas series
    '''
    pred_probs = []
    for i in range(len(trained_classifiers)):
        print('Predicting probabilies with testing set {}'.format(i+1))
        features = testing_splits[i].drop('no_renew_nextpd', axis=1)
        pred_probs.append(pl.predict_target_probability(trained_classifiers[i],
                                                        features))

    return pred_probs

def evaluate_classifiers(pred_probs, testing_splits, seed=None, model_name=None,
                         fig_prefix=None):
    '''
    Prints out evaluations for the trained model using the specified testing
    datasets

    Inputs:
    pred_probs (list of pandas series): list of predicted probabilities
        generated by some classifier; the i-th series of predicted probabilities
        should be associated with the i-th training split
    testing_splits (list of pandas dataframe): the i-th testing split should be
        associated with the i-th series of predicted probabilities
    seed (str): seed used for random process to adjucate ties when translating
        predicted probabilities to predicted classes given some percentile
        threshold
    model_name (str): model name to include in the title of the
        precision/recall curve graph
    fig_name (str): prefix of file name to save the precision/recall curve in;
        if not specified the figure is displayed but not saved
    '''
    table = pd.DataFrame()
    for i in range(len(pred_probs)):
        print('Evaluating predictions with testing set {}'.format(i+1))
        y_actual = testing_splits[i].no_renew_nextpd
        table['Test/Training Set {}'.format(i + 1)], fig =\
            pl.evaluate_classifier(pred_probs[i], y_actual,\
            [0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50], seed=seed,
            model_name=model_name,
            dataset_name='Training/Testing Set # {}'.format(i + 1))
        if fig_prefix is not None:
            plt.savefig(fig_prefix + '_dataset' + str(i + 1) + '.png')
            plt.close()
        else:
            plt.show()
    print(table)

def parse_args(args):
    '''
    Parses dictionary of arguments (typically from the command line) for use by
    the rest of the software

    Inputs:
    args (dict): dict of arguments, typically from the command line; valid keys
        are:
        - 'dataset': path to the Donors' Choose dataset (required)
        - 'features': path to the features config json file (required)
        - 'models': path to the model specs json file (required)
        - 'preprocess': path to the preprocessing config json file (optional)
        - 'seed': numeric seed for tiebreaking (optional)
        - 'save_figs': boolean for wheter figures should be saved or displayed
                       (optional)

    Returns: 6-ple of filepath to dataset (str), pre-procesing specs (dict),
    feature generation specs (dict), model specs (list of dicts), seed (int),
    whether or not to save figures (boolean)
    '''
    dataset_fp = args['dataset']

    if 'preprocess' in args:
        with open(args['preprocess'], 'r') as file:
            preprocess_specs = json.load(file)
    else:
        preprocess_specs = {}

    with open(args['features'], 'r') as file:
        feature_specs = json.load(file)

    with open(args['models'], 'r') as file:
        model_specs = json.load(file)

    seed = args.get('seed', None)

    save_figs = args.get('save_figs', False)

    return dataset_fp, preprocess_specs, feature_specs, model_specs, seed, save_figs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=("Apply machine learning" +
                                                  "pipeline to Donors' Choose Data"))
    parser.add_argument('-d', '--data', type=str, dest='dataset', required=False,
                        help="Path to the Donors' Choose dataset")
    parser.add_argument('-f', '--features', type=str, dest='features',
                        required=True, help="Path to the features config JSON")
    parser.add_argument('-m', '--models', type=str, dest='models',
                        required=True, help="Path to the model specs JSON")
    parser.add_argument('-p', '--preprocess', type=str, dest='preprocess',
                        required=False, help="Path to the preprocessing config JSON")
    parser.add_argument('-s', '--seed', type=int, dest='seed', required=False,
                        help='Random seed for tiebreaking when predicting classes')
    parser.add_argument('--savefigs', dest='save_figs',
                        required=False, action='store_true',
                        help='Save figures instead of displaying them')
    args = parser.parse_args()

    data, preprocess, features, models, seed, save_figs = parse_args(vars(args))
    apply_pipeline(preprocess, features, models, data, seed, save_figs)
