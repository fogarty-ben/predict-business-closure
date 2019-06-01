'''
Class for a ML pipeline

Aya Liu
'''
from __future__ import division
import os
import numpy as np
import pandas as pd
import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pylab as pl
import seaborn as sns
from dateutil import parser
from dateutil.relativedelta import relativedelta
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import data_explore as explore
import data_preprocess as preprocess
from model import *


class Pipeline:
    '''
    Class for a machine learning pipeline
    '''

    LARGE_GRID = { 
                'RF':{'n_estimators': [1,10,100,1000,10000], 'max_depth': [1,5,10,20,50,100], 
                      'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10], 'n_jobs': [-1]},
                'LR': {'penalty': ['l1','l2'], 'C': [0.00001,0.0001,0.001,0.01,0.1,1,10]},
                'ET': {'n_estimators': [1,10,100,1000,10000], 'max_depth': [1,5,10,20,50,100], 
                       'min_samples_split': [2,5,10], 'max_features': ['sqrt','log2'],'n_jobs': [-1]},
                'AB': {'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000,10000]},
                'GB': {'n_estimators': [1,10,100,1000,10000], 'learning_rate' : [0.001,0.01,0.05,0.1,0.5],
                       'subsample' : [0.1,0.5,1.0], 'max_depth': [1,3,5,10,20,50,100]},
                'NB' : {},
                'DT': {'max_depth': [1,5,10,20,50,100], 'min_samples_split': [2,5,10]},
                # 'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],'kernel':['linear']},
                # 'KNN' :{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],
                #         'algorithm': ['auto','ball_tree','kd_tree'], 'n_jobs': [-1]}
                }
    
    SMALL_GRID = { 
                'RF':{'n_estimators': [100, 10000], 'max_depth': [5,50], 'max_features': ['sqrt'],
                      'min_samples_split': [2,10], 'n_jobs':[-1]},
                'LR': {'penalty': ['l1','l2'],'C': [0.001,0.1,1,10]},
                'ET': {'n_estimators': [100, 10000], 'max_depth': [5,50], 'max_features': ['sqrt'],
                       'min_samples_split': [2,10], 'n_jobs':[-1]},
                'AB': {'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [100,10000]},
                'GB': {'n_estimators': [100, 10000], 'learning_rate' : [0.1,0.5],
                       'subsample' : [0.5,1.0], 'max_depth': [5, 50]},
                'NB' : {},
                'DT': {'max_depth': [1,5,20,50], 'min_samples_split': [2, 10]},
                # 'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],'kernel':['linear']},
                # 'KNN' :{'n_neighbors': [1,5,10,25,50,100], 'weights': ['uniform','distance'],
                #         'algorithm': ['auto','ball_tree','kd_tree'], 'n_jobs': [-1]}
                }
        
    TEST_GRID = { 
                'RF':{'n_estimators': [1], 'max_depth': [1], 'max_features': ['sqrt'],
                      'min_samples_split': [10], 'n_jobs': [-1]},
                'LR': {'penalty': ['l1'], 'C': [0.01]},
                'ET': {'n_estimators': [1], 'criterion' : ['gini'], 'max_depth': [1], 
                       'max_features': ['sqrt'],'min_samples_split': [10], 'n_jobs': [-1]},
                'AB': {'algorithm': ['SAMME'], 'n_estimators': [1] },
                'GB': {'n_estimators': [1], 'learning_rate' : [0.1], 'subsample' : [0.5], 'max_depth': [1]},
                'NB' : {},
                'DT': {'max_depth': [1], 'min_samples_split': [10]},
                # 'SVM' :{'C' :[0.01], 'kernel':['linear']},
                # 'KNN' :{'n_neighbors': [5],'weights': ['uniform'], 'algorithm': ['auto'], 'n_jobs': [-1]}
                }    

    def __init__(self):
        '''
        Constructor for a Pipeline.
        '''
        self.df = None
        self.label = None
        self.predictor_sets = []
        self.predictor_combos = []
        self.models = []
        self.train_test_times = []
        self.grid_size = None
        self.paramgrid = None
        self.clfs = {
                    'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
                    'ET': ExtraTreesClassifier(n_estimators=10, n_jobs=-1, criterion='entropy'),
                    'AB': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200),
                    'LR': LogisticRegression(penalty='l1', C=1e5),
                    # 'SVM': svm.SVC(kernel='linear', probability=True, random_state=0),
                    'GB': GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10),
                    'NB': GaussianNB(),
                    'DT': DecisionTreeClassifier(),
                    # 'KNN': KNeighborsClassifier(n_neighbors=3) 
                        }

    def load_clean_data(self, df):
        '''
        Load data into the pipeline after cleaning.
        
        Input:
            df: clean dataframe
        '''
        self.df = df

    def add_predictor_sets(self, predictor_sets, reset=True):
        '''
        predictor_sets: list of lists
        '''
        if reset:
            self.predictor_sets = []

        # update list of predictor sets
        self.predictor_sets.extend(predictor_sets)

        # update feature combinations
        pred_set_combos = get_subsets(self.predictor_sets)
        pred_combos = []
        for v in pred_set_combos:
            merged = list(itertools.chain.from_iterable(v))
            pred_combos.append(merged)
        self.predictor_combos = pred_combos

    #### modeling functions ####

    def run(self, df, time_col, predictor_sets, label, start, end, test_window_years, 
            outcome_lag_years, output_dir, output_filename, grid_size='test', thresholds=[], 
            ks=list(np.arange(0, 1, 0.1)), save_output=True, debug=False):
        '''
        Run the pipeline using temporal cross validation. 

        Inputs:

        predictor_sets: list of lists of predictors

        Output:

        '''
        if debug:
            print('START')
            print('GRID SIZE = {}'.format(grid_size))

        # load data
        self.load_clean_data(df)
        # set outcome
        self.label = label
        # set predictors
        self.add_predictor_sets(predictor_sets, reset=True)
        # set parametergrid for classifiers
        self.grid_size = grid_size
        self.set_paramgrid(grid_size)
        # get cutoff times for train test split
        self.get_train_test_times(start, end, test_window_years, outcome_lag_years)
        # initialize run number and results
        N = 0
        model_results = {}
        # initialize output file
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        output_path = os.path.join(output_dir, output_filename)
        headers = ['model_id', 'N_split', 'i', 'label', 'model_type', 'roc_auc', 'k',
                'precision', 'recall', 'accuracy', 'params', 'predictors']
        pd.DataFrame(columns=headers).to_csv(output_path, index=False)

        if debug:
            print("set up done. output: {}".format(output_path))

        # 1) loop over temporal sets
        for train_start, train_end, test_start, test_end in self.train_test_times:
            model_results[N] = {}
            i = 0

            if debug:
                print('## TRAIN: {} - {}, TEST:{} - {} ##'.format(str(train_start), str(train_end),
                                                    str(test_start), str(test_end)))

            # 2) loop over predictor combos
            for predictor_cols in self.predictor_combos:

                if debug:
                    print('### Predictors: {}'.format(predictor_cols))

                # train test split
                X_train, y_train, X_test, y_test = self.temporal_split(
                                                    time_col, train_start, train_end, 
                                                    test_start, test_end, predictor_cols)
                if debug:
                    print('...train test split done')

                # pre-process training and test sets
                preprocess.preprocess(X_train, y_train)
                preprocess.preprocess(X_test, y_test)

                if debug:
                    print('...pre-processing done')

                # generate features (convert to dummies)
                train_to_concat = [X_train]
                test_to_concat = [X_test]
                for p in predictor_cols:
                    dummies_train, dummies_test = self.generate_feature(p, X_train, X_test)
                    train_to_concat.append(dummies_train)
                    test_to_concat.append(dummies_test)
                X_train = pd.concat(train_to_concat, axis=1)
                X_test = pd.concat(test_to_concat, axis=1)
                X_train.drop(columns=predictor_cols, inplace=True)
                X_test.drop(columns=predictor_cols, inplace=True)

                if debug:
                    print('...feature generation done')

                # 3) loop over classifier types
                for model_type, clf in self.clfs.items():
                    if debug:
                        print('#### {}-{}: {}'.format(N, i, model_type))
                    # 4) loop over parameter combinations
                    for params in ParameterGrid(self.paramgrid[model_type]):
                        if debug:
                            print('{}'.format(params))
                        m = self.build_model(clf, X_train, y_train, X_test, y_test, params, N, i, 
                            model_type, predictor_cols, label, output_dir, output_filename, thresholds, 
                            ks, save_output)
                        model_results[N][i] = m

                        if debug:
                            print('---model results saved---')

                        i += 1
            N += 1

        # store model results
        self.models = model_results

        if debug:
            print('FINISH')

    def build_model(self, clf, X_train, y_train, X_test, y_test, params, N, i, model_type, 
                     predictors, label, output_dir, output_filename, thresholds, ks, save_output):
        '''
        Build and evaluate one classfier Model object.

        Inputs: see self.run() inputs
        Returns: a Model object
        '''
        # initialize
        m = Model(clf, X_train, y_train, X_test, y_test, params, N, i, model_type, 
                  predictors, label, output_dir, thresholds, ks)
        # build
        m.fit_and_predict()
        # evaluate
        m.populate_evalutaions(output_filename)
        m.plot_roc(save_output)
        m.plot_precision_recall_curve(save_output)
        return m

    def generate_feature(self, p, X_train, X_test):
        '''
        p: (str) predictor column name
        X_train, X_test: (dataframe) training and test sets
        '''
        ### get dummies
        # discretize numerical predictors
        if p in preprocess.predictors_to_discretize():
            bins, labels = preprocess.predictors_to_discretize()[p]
            X_train[p] = discretize(X_train, p, bins, labels)
            X_test[p] = discretize(X_test, p, bins, labels)
            # convert to dummies, don't need to add "other" column
            dummies_train = convert_to_dummy(X_train, p, add_other=False)
            dummies_test = convert_to_dummy(X_test, p, add_other=False)

        # convert categorical predictors to dummies
        else: 
            dummies_train = convert_to_dummy(X_train, p, add_other=True)
            dummies_test = convert_to_dummy(X_test, p, add_other=False)

        ### adjust dummies
        # group test-only dummy columns into "other"
        values_not_seen_in_train = dummies_test.columns.difference(dummies_train.columns)
        if len(values_not_seen_in_train) != 0:
            dummies_test['{}_other'.format(p)] = dummies_test[values_not_seen_in_train].sum(axis=1)
        dummies_test.drop(columns=values_not_seen_in_train, inplace=True)

        # add train-only dummy columns to test set
        for c in dummies_train.columns.difference(dummies_test.columns):
            dummies_test[c] = 0

        return dummies_train, dummies_test


    #### temporal cross validation helper functions ####

    def set_paramgrid(self, grid_size):
        '''
        Set parameter grid.

        Input: grid_size: (str) 'large', 'small', or 'test'
        '''
        assert grid_size in ['large', 'small', 'test']
        self.grid_size = grid_size

        if self.grid_size == 'large':
            self.paramgrid = self.LARGE_GRID
        elif self.grid_size == 'small':
            self.paramgrid = self.SMALL_GRID
        else:
            self.paramgrid = self.TEST_GRID

    def temporal_split(self, time_col, train_start, train_end, test_start, test_end, predictor_cols):
        '''
        do one train-test-split according to start/end time
        
        Inputs:
            time_col: (str) time column to split on
            train_start, train_end, test_start, test_end: (datetime) time bound for training and test set
            predictor_cols: (list) predictor column names

        Returns: tuple of X_train, y_train, X_test, y_test
        '''
        # train test split
        train_df = self.df[(self.df[time_col] >= train_start) & (self.df[time_col] <= train_end)]
        test_df = self.df[(self.df[time_col] >= test_start) & (self.df[time_col] <= test_end)]

        # select features cols only
        X_train = train_df[predictor_cols]
        X_test = test_df[predictor_cols]

        # generate outcome

        return X_train, y_train, X_test, y_train

    def get_train_test_times(self, start, end, test_window_years, outcome_lag_years):
        '''
        start: (datetime) start time of all data
        end: (datetime) end time of all data
        test_window_years: (int/float) time length of a test set in months
        outcome_lag_years: (int/foat) lag needed to get the outcome in days

        '''
        results = []

        # initial train/test time cutoffs
        train_start = start
        train_end = train_start + \
                    relativedelta(years=+test_window_years) - \
                    relativedelta(days=+1)
        test_start = train_end + \
        			relativedelta(years=+outcome_lag_years) + \
        			relativedelta(days=+1)
        test_end =  test_start + \
                    relativedelta(years=+test_window_years) - \
                    relativedelta(days=+1)

        while test_end <= end - relativedelta(years=+outcome_lag_years):
                # save times
                results.append((train_start, train_end, test_start, test_end))

                # increment time (train_start stays the same)
                test_start = test_end + relativedelta(days=+1)
                test_end = test_start + \
                            relativedelta(years=+test_window_years) - \
                            relativedelta(days=+1)
                train_end = test_start - \
                			relativedelta(years=+outcome_lag_years) - \
                			relativedelta(days=+1)
               
        # # adjust test_end for the last test set:
        # test_end = end - relativedelta(years=+outcome_lag_years)
        # results.append((train_start, train_end, test_start, test_end))
        self.train_test_times = results


### model comparison functions ###
def compare_model_precisions_at_k(evaluations_filepath, k,  Ns=[], model_types=[], model_ids=[],
    save_output=True, output_filepath=None):
    '''
    compare precision at k (%) population for different models
    
    Inputs:
        evaluations_filepath: (str) filepath for evaluations.csv generated by pipeline 
        k (float between 0 and 0.1): percentage of population labeled as 1
        Ns, model_types, model_ids: (optional lists) criteria to filter for models to compare.
                                    Default=[]
        save_output: (optional bool) whether to save plot
        output_filepath: (optional str)
    '''
    ev = pd.read_csv(evaluations_filepath)
    
    # filtering for models to compare
    if Ns:
        ev = ev[ev['N'].isin(Ns)]
    if model_types:
        ev = ev[ev['model_type'].isin(model_types)]
    if model_ids:
        ev = ev[ev['model_id'].isin(model_ids)]

    # plot
    plt.clf()
    fig = sns.barplot(x=ev[ev['k']==k]['model_id'], y=ev[ev['k']==0.1]['precision'])
    if save_output:
        plt.savefig(output_filepath)


def compare_model_recalls_at_k(evaluations_filepath, k,  Ns=[], model_types=[], model_ids=[],
    save_output=True, output_filepath=None):
    '''
    compare recall at k (%) population for different models

    Inputs:
        evaluations_filepath: (str) filepath for evaluations.csv generated by pipeline 
        k: (float between 0 and 0.1) percentage of population labeled as 1
        Ns, model_types, model_ids: (optional lists) criteria to filter for models to compare.
                                    Default=[]
        save_output: (optional bool) whether to save plot
        output_filepath: (optional str)
    '''
    ev = pd.read_csv(evaluations_filepath)

    # filtering for models to compare
    if Ns:
        ev = ev[ev['N'].isin(Ns)]
    if model_types:
        ev = ev[ev['model_type'].isin(model_types)]
    if model_ids:
        ev = ev[ev['model_id'].isin(model_ids)]

    # plot
    plt.clf()
    fig = sns.barplot(x=ev[ev['k']==k]['model_id'], y=ev[ev['k']==0.1]['recall'])
    if save_output:
        plt.savefig(output_filepath)    


def compare_model_aucs(evaluations_filepath, Ns=[], model_types=[], model_ids=[],
    save_output=True, output_filepath=None):
    '''
    compare AUC scores for different models.

    Inputs:
        evaluations_filepath: (str) filepath for evaluations.csv generated by pipeline 
        Ns, model_types, model_ids: (optional lists) criteria to filter for models to compare.
                                    Default=[]
        save_output: (optional bool) whether to save plot
        output_filepath: (optional str)
    '''
    ev = pd.read_csv(evaluations_filepath)

    # filtering for models to compare
    if Ns:
        ev = ev[ev['N'].isin(Ns)]
    if model_types:
        ev = ev[ev['model_type'].isin(model_types)]
    if model_ids:
        ev = ev[ev['model_id'].isin(model_ids)]

    # plot
    x = list(ev.groupby('model_id').groups)
    y = ev.groupby('model_id').agg('mean')['roc_auc']   
    plt.clf()
    fig = sns.barplot(x, y)

    if save_output:
        plt.savefig(output_filepath)


### util functions for pipeline ###

def read_csv(data_filepath, coltypes=None, parse_dates=None):
    '''
    Read csv file into a dataframe.
        
    Inputs:
        data_filepath: (str) data filepath
        coltypes: (optional dict) column, data type pairs. default=None.
        parse_dates: (optional list) columns to parse as datetime 
                     objects. default=None.
    '''
    return pd.read_csv(data_filepath, dtype=coltypes, parse_dates=parse_dates)


def explore_data(df, label, num_vars=None, cat_vars=None):
    '''
    Explore distribution of numerical and categorical variables
    and missing data. Show tables and plots.

    Inputs: 
        num_vars (optional list): column names of numerical vars
        cat_vars (optional list): column names of categorical vars
    '''
    if num_vars:
        explore.explore_num_vars(label, num_vars, df)
    if cat_vars:
        explore.explore_cat_vars(label, cat_vars, df)
    explore.summarize_missing(df)


def get_subsets(l):
    '''
    Get all subsets of the list l

    Input: l (list)
    Returns: subsets (list of lists)
    '''
    subsets = []
    for i in range(1, len(l) + 1):
        for combo in itertools.combinations(l, i):
            subsets.append(list(combo))
    return subsets

def discretize(df, p, bins=3, labels=None):
    '''
    Inputs:
        bins: (optional int, sequence of scalars, or pandas.IntervalIndex)
              the criteria to bin by. Default=3
        labels: (optional array or bool) specifies the labels for the 
                returned bins. Default=None
    '''
    return pd.cut(df[p], bins=bins, labels=labels)


def convert_to_dummy(data, col, add_other):
    '''
    Convert a list of categorical variables to dummy variables.

    Inputs:
        data: (dataframe) the dataframe
        col: column to convert to dummies 
        add_other: (bool) whether to add an "other" dummy column

    Returns: a dataframe containing only dummy columns
    '''
    dummies = pd.get_dummies(data[col], prefix=col, dummy_na=True)
    if add_other:
        col_other = '{}_other'.format(col)
        if col_other not in dummies.columns:
            dummies[col_other] = 0
    return dummies
