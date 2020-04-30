'''
Predicting Business Closures

Ben Fogarty
Parth Khare
Aya Liu

Harris School of Public Policy, University of Chicago
CAPP 30254: Machine Learning for Public Policy
Prof. Rayid Ghani

12 June 2019
'''

import argparse
import datetime
import json
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import load_data
import pipeline_library as pl

def apply_pipeline(preprocessing, features, models, dataset=None, seed=None,
                   save_figs=False, save_preds=False, save_eval=False):
    '''
    Applies the pipeline library to predicting if a licensed business on a
    prediction date will renew its license in the next two-year period.

    Inputs:
    preprocessing (dict): preprocessing specifications as a dictionary of
        keyword arguments to pass to the preprocess_data function
    features (dict): feature generation specifiations as a dictionary of keyword
        arguments to pass to the generate_features function in
        pipeline_library
    models (dict): list of model specification dictionaries
    dataset (str): path to the pickle file if using frozen dataset
    seed (str): seed used for random process to adjucate ties when translating
        predicted probabilities to predicted classes given some percentile
        threshold
    save_figs (bool): if true, figures are saved instead of displayed
    save_preds (bool): if true, predictions on test sets are saved
    save_eval (bool): if true, evaluation metrics are saved
    '''
    if dataset is None:
        df = load_data.get_lcs_data()
    else:
        with open(dataset, 'rb') as file:
            df = pickle.load(file)

    print('Generating training/testing splits...')
    training_splits, testing_splits = pl.create_temporal_splits(df, 'pred_date',
                                                                {'years': 2},
                                                                gap={'years': 2},
                                                                start_date="2006-01-01",
                                                                end_date='2016-01-01')

    print('Preprocessing data and generating features...')
    for i in range(len(training_splits)):
        training_splits[i] = pl.preprocess_data(training_splits[i], **preprocessing)
        testing_splits[i] = pl.preprocess_data(testing_splits[i], **preprocessing)

        training_splits[i], testing_splits[i] = generate_features(training_splits[i],
                                                                  testing_splits[i],
                                                                  **features)

        print('_' * 20 + '\nTesting set #{}\n'.format(i + 1) + '_' * 20)
        print('no_renew_nextpd baseline: {}'.format(np.mean(testing_splits[i].no_renew_nextpd)))
        print('number of observations: {}'.format(len(testing_splits[i])))

    for i in range(len(models)):
        model = models[i]
        print('-' * 20 +  '\nModel Specifications\n' + str(model) + '\n' + '_' * 20)
        print('Start time: {}\n'.format(datetime.datetime.now()))
        model_name = model.get('name', 'model-{}'.format(i + 1))
        trained_classifiers = train_classifiers(model, training_splits)
        print('\n')
        pred_probs = predict_probs(trained_classifiers, testing_splits)
        if save_preds:
            for i, prediction in enumerate(pred_probs):
                testing_splits[i]['pred_class_10%'] = pl.predict_target_class(pred_probs[i], 0.1,
                                                                              seed=seed)
                testing_splits[i].to_csv(model_name + '_set-{}_pred_probs.csv'.format(i + 1),
                                         index=False)
                testing_splits[i] = testing_splits[i].drop('pred_class_10%', axis=1)
        print('\n')
        if save_figs:
            eval_tbl = evaluate_classifiers(pred_probs, testing_splits, seed,
                                            model_name, fig_prefix=model_name)
        else:
            eval_tbl = evaluate_classifiers(pred_probs, testing_splits, seed,
                                            model_name)
        print(eval_tbl.to_string())
        if save_eval:
            eval_tbl.to_csv(model_name + '_set-{}_eval.csv'.format(i + 1))

        print('\nEnd time: {}'.format(datetime.datetime.now()))

        return trained_classifiers

def generate_features(training, testing, n_ocurr_cols, scale_cols, bin_cols,
                      dummy_cols, iter_dummy_cols, binary_cut_cols,
                      duration_cols, interaction_cols, drop_cols):
    '''
    Generates categorical, binary, and scaled features. While features are
    generated for the training data independent of the testing data, features
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
    - Duration columns (automatically scaled) (new column with original name +
      '_duration')
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
    binary_cut_cols (dict of dicts): each key is the name of a column to cut
        into two groups based on some threshold and each value is a dictionry
        of arguments to pass to the cut_binary function in pipeline_library
        (must contain a value for threshold, or_equal_to parameter is optional)
    duration_cols (list of tuples of column names): first column is name of
        column containg start date, second column is name of column containing
        end dates
    interaction_cols (list of n-ples of col names): each tuple contains names of
        columns to interact with one another
    drop_cols (list of strs): names of columns to drop

    Returns: tuple of pandas dataframe, the training and testing datasets after
        generating features
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
        max_training = max(training[start_col + '-' + end_col + "_duration"])
        min_training = min(training[start_col + '-' + end_col + "_duration"])
        training.loc[:, start_col + '-' + end_col + "_duration_scale"] =\
            pl.scale_variable_minmax(training[start_col + '-' + end_col + "_duration"],
                                     a=max_training, b=min_training)
        testing.loc[:, start_col + '-' + end_col + "_duration_scale"] =\
            pl.scale_variable_minmax(testing[start_col + '-' + end_col + "_duration"],
                                     a=max_training, b=min_training)
        training = training.drop(start_col + '-' + end_col + "_duration", axis=1)
        testing = testing.drop(start_col + '-' + end_col + "_duration", axis=1)

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
    model (dict): specifications for the classifier (see pipeline library
        function generate_classifier for more information)
    training (list of pandas dataframe): a list of training datasets

    Returns: 2-D list of trained sklearn classifiers
    '''
    classifiers = []
    fi_available = model['model'] in ['rf', 'dt', 'boosting', 'bagging', 'lr',
                                      'svm']
    for i in range(len(training)):
        print('Building with training set {}'.format(i + 1))
        features = training[i].drop('no_renew_nextpd', axis=1)
        target = training[i].no_renew_nextpd
        classifiers.append(pl.generate_classifier(features, target, model))
        feature_importance = pl.get_feature_importance(features, classifiers[-1],
                                                       model)
        if fi_available: # display feature importance
            print(feature_importance.sort_values('Importance', ascending=False)\
                                    .head(15)\
                                    .to_string())
        else:
            print(feature_importance)
        print('\n')

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
    datasets.

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
    fig_prefix (str): prefix of file name to save the precision/recall curve in;
        if not specified the figure is displayed but not saved

    Returns a pandas dataframe with evaluation metrics
    '''
    table = pd.DataFrame()
    for i in range(len(pred_probs)):
        print('Evaluating predictions with testing set {}'.format(i+1))
        y_actual = testing_splits[i].no_renew_nextpd
        table['Test/Training Set {}'.format(i + 1)], fig =\
            pl.evaluate_classifier(pred_probs[i], y_actual,\
                                   [0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50],
                                   seed=seed, model_name=model_name,
                                   dataset_name='Training/Testing Set # {}'.format(i + 1),
                                   tie_breaker='pessimistic')
        if fig_prefix is not None:
            plt.savefig(fig_prefix + '_dataset' + str(i + 1) + '.png')
            plt.close()
        else:
            plt.show()

    return table

def parse_args(args):
    '''
    Parses dictionary of arguments (typically from the command line) for use by
    the rest of the software

    Inputs:
    args (dict): dict of arguments, typically from the command line; valid keys
        are:
        - 'dataset': path to the dataset (required)
        - 'features': path to the features config json file (required)
        - 'models': path to the model specs json file (required)
        - 'preprocess': path to the preprocessing config json file (optional)
        - 'seed': numeric seed for tiebreaking (optional)
        - 'save_figs': boolean for wheter figures should be saved or displayed
                       (optional)
        - 'save_pred': boolean for whether predictions should be output to csv
                       (optional)
        - 'save_eval': boolean for whether evaluation metrics should be output
                       to csv (optional)

    Returns: 6-ple of filepath to dataset (str), pre-procesing specs (dict),
    feature generation specs (dict), model specs (list of dicts), seed (int),
    whether or not to save figures (boolean), whether or not to save evaluation
    metrics (boolean)
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

    save_preds = args.get('save_preds', False)

    save_eval = args.get('save_eval', False)

    return dataset_fp, preprocess_specs, feature_specs, model_specs, seed,\
           save_figs, save_preds, save_eval


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=("Apply machine learning" +
                                                  "pipeline to business license"))
    parser.add_argument('-d', '--data', type=str, dest='dataset', required=False,
                        help=("Optional path to the business dataset pickle so that" +
                              " the dataset doesn't have to be redownloaded"))
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
    parser.add_argument('--savepreds', dest='save_preds',
                        required=False, action='store_true',
                        help='Save predictions to file')
    parser.add_argument('--saveeval', dest='save_eval',
                        required=False, action='store_true',
                        help='Save evaluations to file')
    args = parser.parse_args()

    data, preprocess, features, models, seed, save_figs, save_preds, save_eval = parse_args(vars(args))
    apply_pipeline(preprocess, features, models, data, seed, save_figs,
                   save_preds, save_eval)
