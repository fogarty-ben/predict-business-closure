'''
Aya Liu
'''

from IPython.display import display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# classifiers
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, \
     GradientBoostingClassifier, BaggingClassifier
# model selection
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid
# metrics
from sklearn import metrics
from sklearn.metrics import roc_curve  
from sklearn.metrics import roc_auc_score  
import explore_utils as explore


MODELS_TO_BUILD = {
    'DecTree': {
        'clf': DecisionTreeClassifier(),
        'paramgrid': {'max_depth': [10, 25, 50]}
    },
    'KNN': {
        'clf': KNeighborsClassifier(),
        'paramgrid': {'n_neighbors': [10, 20], 
                      'weights': ['uniform', 'distance']
                      } 
    },
    'LogReg': {
        'clf': LogisticRegression(),
        'paramgrid': {'penalty': ['l1', 'l2']} 
    },      
    'RanFor': {
        'clf': RandomForestClassifier(),
        'paramgrid': {'n_estimators': [50, 100, 200]} 
    },
    'Boosting': {
        'clf': GradientBoostingClassifier(),
        'paramgrid': {'n_estimators': [50, 100, 200]} 
    },
    'Bagging': {
        'clf': BaggingClassifier(),
        'paramgrid': {'n_estimators': [10, 50]}
    },
    
}

# constants for plotting
LINE_STYLES = ['solid', 'dashed', 'dashdot', 'dotted']
LINE_WIDTHS = [0.5, 2, 3.]


########## read ##########

def read_data(data_file, coltypes=None, parse_dates=None):
	'''
	Read csv file into a dataframe.

	'''
	df = pd.read_csv(data_file, dtype=coltypes, parse_dates=parse_dates)
	return df


########## explore ##########

def explore_data(group, num_vars, cat_vars, data):
    '''
    Explore distribution of numerical and categorical variables.
    Show tables and plots.

    '''
    explore.explore_num_vars(group, num_vars, data)
    explore.explore_cat_vars(group, cat_vars, data)
    explore.summarize_missing(data)


########## pre-process ##########

def fill_na_with_median(data):
    '''
    Fill all columns with NA values with median values of those columns

    Inputs: a dataframe
    Returns: a dataframe with no NA values

    '''
    cols_na = {col: data[col].median() for col in data.columns 
               if data[col].isna().any()}
    return data.fillna(cols_na)


########## generate features ##########

def discretize(varname, data, bins, labels=None):
    '''
    Convert a continous variable to a categorical variable.

    Inputs:
        varname: (str) name of the continous variable
        data: (dataframe) the dataframe
        bins: (int, sequence of scalars, or pandas.IntervalIndex)
              the criteria to bin by. 
        (Optional) labels: (array or bool) specifies the labels for the 
                           returned bins. 
    Returns: a pandas Series of the categorical variable

    '''
    return pd.cut(data[varname], bins=bins, labels=labels)


def convert_to_dummy(data, cols_to_convert, dummy_na=True):
    '''
    Convert a list of categorical variables to dummy variables.

    Inputs:
        cols_to_convert: (list) list of variable names
        data: (dataframe) the dataframe
        dummy_na: (bool)
    Returns: a dataframe containing dummy variables of cols_to_convert

    '''
    return pd.get_dummies(data, dummy_na=dummy_na, columns=cols_to_convert)


########## build classifiers ##########

##### main function
def build_temporal_classifiers(models_to_build, sets, feat_cols, 
                               plot_pred_scores=False):
    '''
    Build classifier for each train-test split
    
    '''
    results = []
    for s in sets: 
        print("Train: Start", s['train_start'], "| End", s['train_end'])
        print("Test: Start", s['test_start'], "| End", s['test_end'])
        print("--------------------------------")

        models_from_s = build_classifiers(
            models_to_build, s['x_train'], s['x_test'], s['y_train'], 
            s['y_test'], feat_cols, plot_pred_scores, s['train_start'], 
            s['train_end'], s['test_start'], s['test_end']
            )
        results += models_from_s

    print("Finished all temporal splits.")
    return results


def build_classifiers(models_to_build, x_train, x_test, y_train, y_test,
                      feat_cols, plot_pred_scores=False, train_start=None, 
                      train_end=None, test_start=None, test_end=None):
    '''
    '''
    models_built = []

    for model_name in models_to_build:
        print("######{}######".format(model_name))

        paramgrid = ParameterGrid(models_to_build[model_name]['paramgrid'])
        
        for params in paramgrid:
            print("\nBuilding {}: {}...".format(model_name, params))

            # build 
            clf = models_to_build[model_name]['clf'].set_params(**params)
            # train
            clf = clf.fit(x_train, y_train)
            # test
            pred_scores = clf.predict_proba(x_test)
            # plot pred scores
            if plot_pred_scores:
                plt.hist(pred_scores[:,1])
                plt.title("Predicted scores on test set")
                plt.show()
            # store model results
            model_built_result = create_built_model_result(
                clf, model_name, params, y_test, pred_scores, feat_cols, 
                train_start, train_end, test_start, test_end)
            models_built.append(model_built_result)

            print("Built\n")

    return models_built

##### utils

def get_feature_cols(target, data_dummies):
    '''
    feature columns must come after the target variable column
    '''

    # drop NaN dummies for non-NaN variables
    data_dummies = data_dummies.loc[:, (data_dummies != 0).any(axis=0)]

    # select a vector of features
    cols = data_dummies.columns.to_list()
    feat_cols = cols[cols.index(target)+1:]

    return feat_cols


def temporal_train_test_split(target, feat_cols, data, date_col, n_splits,
                              show_sets=False):
    '''
    split time series data into training and testing sets

    '''
    # sort data by date
    df_sorted = data.set_index(date_col).sort_index()

    # split dataset into features x and target variable y
    x = df_sorted[feat_cols]
    y = df_sorted[target]

    # do temporal split
    tscv = TimeSeriesSplit(n_splits=n_splits)
    sets = []
    for train_index, test_index in tscv.split(x):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # get start/end date for training and testing sets
        train_start = y_train.index[0]
        train_end = y_train.index[-1]
        test_start = y_test.index[0]
        test_end = y_test.index[-1]

        sets.append(
            {'x_train': x_train, 'x_test': x_test,
             'y_train': y_train, 'y_test': y_test,
             'train_start': train_start, 'train_end': train_end,
             'test_start': test_start, 'test_end': test_end
            })

    # show train and test sets
    if show_sets:
        for i, s in enumerate(sets):
            print("Set", i)
            print("Train: Start", s['train_start'], "| End", s['train_end'])
            print("Test: Start", s['test_start'], "| End", s['test_end'])
            print("---")

    return sets


def fit_and_predict(clf, x_train, y_train, x_test, plot=False):
    '''
    '''
    clf = clf.fit(x_train, y_train) # train
    pred_scores = clf.predict_proba(x_test) # test

    if plot:
        plt.hist(pred_scores[:,1])
        plt.title("Predicted scores on test set")
        plt.show()
    return  pred_scores


def create_built_model_result(clf, model_name, params, y_test, 
        pred_scores, feat_cols, train_start=None, train_end=None, 
        test_start=None, test_end=None):

    return {'model_name': model_name, 
              'model': clf,
              'params': params,
              'y_test': y_test,
              'pred_scores': pred_scores,
              'feat_cols': feat_cols,
              'train_start': train_start,
              'train_end': train_end,
              'test_start': test_start,
              'test_end': test_end
              }


########## evaluate models ##########

def evaluate_models(models, metrics, population_percent=None,
                    thresholds=None, roc=True):
    '''
    calculate evaluation metrics for a collection of models
    models: dict returned by build_classifiers()
    '''
    models_w_metrics = []
    for m in models:
        # add metrics to model dictionary
        metrics_results = get_model_eval_metrics(
                          m['y_test'], m['pred_scores'], metrics,
                          population_percent, thresholds)
        new_m = dict(m, **metrics_results)

        if roc:
            # add AUC score and ROC curve to model dictionary
            auc = roc_auc_score(m['y_test'], m['pred_scores'][:,1]) 
            fpr, tpr, roc_thresholds = roc_curve(
                                           m['y_test'], m['pred_scores'][:,1])
            new_m['auc'] = auc
            new_m['fpr'] = fpr
            new_m['tpr'] = tpr
            new_m['roc_thresholds'] = roc_thresholds

        models_w_metrics.append(new_m)

    return models_w_metrics


def get_model_eval_metrics(y_test, pred_scores, metrics, 
    population_percent=None, thresholds=None, show_metrics=False):

    # check parameters    
    if (thresholds == None) and (population_percent == None):
            raise Exception('ValueError: must have thresholds or \
                            population_percent')
    elif thresholds and population_percent:
            raise Exception('Cannot have both thresholds and \
                            population_percent')
    elif population_percent:
            # calculate score thresholds for top p% of population
            thresholds = []
            for p in population_percent:
                thresholds.append(np.percentile(pred_scores, (1-p)*100))

    # initialize dict for metric scores
    results = {'population_percent': population_percent,
               'score_threshold': thresholds}

    # calcualte evaluation metrics at various thresholds 
    for m in metrics:
        name = m.__name__[:-6]
        scores = []

        for k in thresholds:
            # assign classification at threshold k
            pred_label = [1 if x[1] > k else 0 for x in pred_scores]
            num_pred_1 = sum(pred_label)
            # calculate evaluation metric at threshold k
            score_at_k =  m(y_true=y_test, y_pred=pred_label)
            scores.append(score_at_k)
        results[name] = scores

    # calculate baseline: % of actual 1s in test data
    bl_perc = sum(y_test)/len(y_test)
    results['baseline'] = [bl_perc] * len(thresholds)

    if show_metrics: # display metrics as a table/dataframe
        print("Baseline: The true number of YES in test data is {}/{}\
              ({:.2f}%)".format(sum(y_test), len(y_test), 100.*bl_perc))
        df = pd.DataFrame(results)
        display(df)

    return results


def plot_roc_curve(fpr, tpr):

    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()


def organize_models_by_set_by_type(models, n_sets):
    '''
    models: list of dicts
    n_sets: int, number of temporal sets

    '''
    # split models built into temporal sets 
    by_set = []
    inc = int(len(models) / n_sets)
    lb, ub = (0, inc)
    for i in range(n_sets):
        by_set.append(models[lb:ub])
        lb += inc
        ub += inc
    
    # reorganize models within each set by type
    by_set_by_type={}
    for j, mbs in enumerate(by_set):
        by_type = {}
        for m in mbs:
            if m['model_name'] not in by_type:
                by_type[m['model_name']] = [m]
            else:
                by_type[m['model_name']].append(m)
        by_set_by_type[str(j+1)] = by_type

    return by_set_by_type


def compare_roc_curves_within_type(by_set_by_type):
    '''
    '''
    sns.set_style('darkgrid')
    # create a figure for each temporal set
    for k, temp_set in by_set_by_type.items():
        fig = plt.figure(figsize=(42,7))
        fig.suptitle("ROC Curve - Temporal set {}".format(k), fontsize=20)

        # create a subplot for each model type within one temporal set:
        i = 1
        for mtype, models in temp_set.items(): 
            ax = fig.add_subplot(1, 6, i)
            ax.set_title(mtype)
            i += 1

            # plot ROC curves for each model under one model type on subplot
            for m in models: 
                plt.plot(m['fpr'], m['tpr'], label=str(m['params']).strip('{}'))
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.legend()

        plt.show()


def compare_roc_curves_across_types(by_set_by_type):
    '''
    '''
    sns.set_style('darkgrid')
    # create a figure for each temporal set
    for k, temp_set in by_set_by_type.items():
        fig = plt.figure(figsize=(8,6))
        fig.suptitle("ROC Curve - Temporal set {}".format(k), fontsize=20)
        ax = fig.add_subplot(111) 

        # plot each model's metric
        for i, (_, models) in enumerate(temp_set.items()): 
            # set line style/width for this classifier type
            ls = LINE_STYLES[(i % len(LINE_STYLES))]
            lw = LINE_WIDTHS[(i % len(LINE_WIDTHS))]

            for m in models: 
                plt.plot(m['fpr'], m['tpr'], linestyle=ls, linewidth=lw, 
                         label=str(m['params']).strip('{}'))
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.legend(bbox_to_anchor=(1.6, 1.05))


def lineplot_metric_at_k_perc_pop(model, key, y_range, ls='solid', lw=1.0,
                                  remove_leading_zeros=False, show_baseline=False):

    if remove_leading_zeros:
        y = np.trim_zeros(model[key], 'f')
        num_rm = len(model[key]) - len(y)
        x_ub = len(model['population_percent']) - num_rm
        x = model['population_percent'][:x_ub]
    else: 
        y = model[key] 
        x = model['population_percent']

    plt.plot(x, y, linestyle=ls, linewidth=lw, label="{}, {}".format(
                   model['model_name'], str(model['params']).strip('{}')))
    plt.yticks(y_range)
    plt.legend(bbox_to_anchor=(1.6, 1.0))


def compare_metrics_across_types(by_set_by_type, metric_name, y_range, figsize=(6,4)):
    '''
    '''
    # create a plot for each temporal set:

    for k, temp_set in by_set_by_type.items():
        fig = plt.figure(figsize=figsize)
        fig.suptitle("{} - Temporal set {}".format(metric_name, k), fontsize=20)
        ax = fig.add_subplot(111)

        # plot each model's metric
        for i, (_, models) in enumerate(temp_set.items()): 
            ls = LINE_STYLES[(i % len(LINE_STYLES))]
            lw = LINE_WIDTHS[(i % len(LINE_WIDTHS))]

            # set line style/width for this classifier type
            for m in models: 
                remove_leading_zeros = (metric_name == "precision")
                lineplot_metric_at_k_perc_pop(
                    m, metric_name, y_range, ls, lw,
                    remove_leading_zeros, show_baseline=False)



def plot_precision_recall(model, show_baseline=True, figsize=(8,6)):
    '''
    Shortcut function to plot precision and recall with x-axis of the top 
    % population that are classified as 1.

    '''
    x = 'population_percent'
    y_list = ['precision', 'recall']
    if show_baseline:
        y_list.append('baseline')

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    for y in y_list:
        plt.plot(model[x], model[y], label=y)
    
    plt.legend()
    plt.title("Precision-Recall at k% Population")
    plt.show()





