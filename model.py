'''
Class for one classifier model 

ML Pipeline
Aya Liu
'''

from __future__ import division
import os
import csv
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pylab as pl
from sklearn import model_selection, metrics
from sklearn.metrics import roc_auc_score, roc_curve, auc, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, accuracy_score


class Model:

    def __init__(self, clf, X_train, y_train, X_test, y_test, params, N, iteration, model_type, 
                 predictors, label, output_dir, thresholds=[], ks=[]):
        '''
        Constructor for a Model.

        Inputs: 
            clf: classifier object
            X_train, y_train: (pd dataframes) training data
            X_test, y_test: (pd dataframes) testing data
            params: (dict) {parameter: value} pairs
            N: (int)
            iteration: (int)
            model_type: (str) model type acronym
            predictors:
            label:
            output_dir: output directory
            thresholds: (optional list of floats between 0 and 1) probability score thresholds. 
                        Default=[]
            ks: (optional list of floats between 0 and 1) percentages of population labeled as 1. 
                        Default=[]

        '''
        # inputs
        self.clf = clf
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.params = params
        self.N = N
        self.iteration = iteration
        self.predictors = predictors
        self.model_type = model_type
        self.label = label
        self.thresholds = thresholds
        self.ks = ks
        # results
        self.y_scores = None
        self.roc_auc = None
        self.precision = []
        self.recall = []
        self.accuracy = []
        self.importances = None
        # output
        self.output_dir = output_dir

    ######## Build ########
    def fit_and_predict(self, debug=False):
        '''
        Runs a model with params.

        Input: 
            debug: (bool)
        '''
        # set model parameters
        self.clf.set_params(**self.params)
        if debug:
            print("set params")
            print(self.clf)

        # fit and predict
        self.y_scores = self.clf.fit(self.X_train, self.y_train).predict_proba(self.X_test)[:,1]
        if debug:
            print("generated prediction scores")

        # store feature importances
        if self.model_type in ['RF', 'ET', 'AB', 'GB', 'DT']:
            self.importances = self.clf.feature_importances_
            if debug:
                print("stored feature importances")
        elif self.model_type in ['SVM', 'LR']:
            self.importances = self.clf.coef_[0]
            if debug:
                print("stored feature importances")        

    ######## Evaluate ########
    def populate_evalutaions(self, output_filename="evalutaions.csv", debug=False):
        '''
        Populate evaluation results to file and update model attributes.

        Input:
            output_filepath: (str) filepath to store model evaluation results
            debug: (bool)
        '''

        # calculate and store auc score
        self.roc_auc = roc_auc_score(self.y_test, self.y_scores)

        # determine if cutoff type is k or probability threshold
        if self.ks != []:
            cutoffs = self.ks
            if debug:
                print("evaluate at k % population:\n")
                cutoff_type = "k"

        elif self.thresholds != []:
            cutoffs = self.thresholds
            if debug:
                print("evaluate at probability thresholds:\n")
                cutoff_type = "threshold"

        # calculate and save other metrics (precision, recall, accuracy) 
        # at each cutoff level
        for cutoff in cutoffs:
            if debug:
                print("- {}={}:".format(cutoff_type, cutoff))

            # calculate and store metrics
            precision, recall, accuracy = self.calculate_metrics_at_cutoff(cutoff)
            self.precision.append(precision)
            self.recall.append(recall)
            self.accuracy.append(accuracy)
            if debug:
                print("\tcalculated evaluation metrics at cutoff")

            # write results to file
            self.model_eval_to_file(cutoff=cutoff, metrics=(precision, recall, accuracy),
                                    output_filename=output_filename)
            if debug:
                print("\twrote evalutaion results to file")

    def calculate_metrics_at_cutoff(self, cutoff, debug=False):
        '''
        Inputs:
            cutoff: (float) a cutoff threshold between [0, 1]
            debug: (bool)

        Returns: tuple of precision, recall, accuracy
        '''

        # if k list is specified for model, calculate metrics at the cutoff % of population       
        if self.ks != []:
            precision = self.precision_at_k(self.y_test, self.y_scores, cutoff)
            recall = self.recall_at_k(self.y_test, self.y_scores, cutoff)
            accuracy = self.accuracy_at_k(self.y_test, self.y_scores, cutoff)
            if debug:
                print("calculated precision, recall, and accuracy at {}% population".format(cutoff*100))

        # if k list is not specified and threshold list is specified, calculate metrics at the 
        # cutoff probability threshold
        elif self.thresholds != []:
            precision = self.precision_at_threshold(self.y_test, self.y_scores, cutoff)
            recall = self.recall_at_threshold(self.y_test, self.y_scores, cutoff)
            accuracy = self.accuracy_at_threshold(self.y_test, self.y_scores, cutoff)
            if debug:
                print("calculated precision, recall, and accuracy at {} probability threshold".format(
                      cutoff))

        # if neither k nor threshold list is specified, do nothing
        else:
            raise Exception("Neither ks nor thresholds are specified for model. No other metrics are calculated.")

        return precision, recall, accuracy

    def model_eval_to_file(self, cutoff, metrics, output_filename='evaluations.csv', mode='a'):
        '''
        Writes evaluation metrics to file.

        Input:
            cutoff: (float between 0 and 1) threshold or k for metric
            metrics: (tuple of floats) precision, recall, and accuracy scores
            output_filename: (optional str): evaluation output filename. default='evaluations.csv'
            mode: (optional str): file handling mode, 'a' for addition, 'w' for writing. default='a'

        File Columns:
        model_id( N-iteration), label, model_type, iteration, AUC, cutoff (k or threshold), 
        precision, recall, accuracy, params
        '''
        precision, recall, accuracy = metrics
        output_filepath = os.path.join(self.output_dir, output_filename)
        row = ["{}-{}".format(self.N, self.iteration), self.N, self.iteration, self.label, 
               self.model_type, self.roc_auc, cutoff, precision, recall, accuracy, self.params, 
               self.predictors]
       
        with open(output_filepath, mode) as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
            csvFile.close()

        #     result = '"{0}-{1}", "{2}", "{3}", "{4}", "{5}", "{6}", "{7}", "{8}", "{9}", "{10}"\n'.format(
        #         self.N, self.iteration, self.label, self.model_type, self.roc_auc, cutoff,
        #         precision, recall, accuracy, self.params, self.predictors)
        #     f.write(result)

    def clear_metrics(self, filepath_to_remove=None):
        '''
        Clear precision, recall, and accuracy attributes.

        Input: 
            filepath_to_remove: (optional str) path of any evaluation file to delete
        '''
        self.precision = []
        self.recall = []
        self.accuracy = []
        if filepath_to_remove:
            os.remove(filepath_to_remove)

    def clear_cutoffs(self):
        '''
        Clear thresholds and ks.
        '''
        self.thresholds = []
        self.ks = []

    def set_thresholds(self, thresholds):
        '''
        Set list of thresholds to evaluate Model on.

        Input: 
            thresholds: list of floats between 0 and 1

        '''
        self.thresholds = thresholds

    def set_ks(self, ks):
        '''
        Set list of k's to evaluate Model on.

        Input: 
            ks: list of percentages of population between 0 and 1
        '''
        self.ks = ks


    def accuracy_at_k(self, y_true, y_scores, k):
        '''
        Accuracy at k (%) population labeled as 1
        '''
        # sort rows by scores
        y_scores_sorted, y_true_sorted = joint_sort_descending(np.array(y_scores), np.array(y_true))
        # label predictions at k 
        preds_at_k = self.generate_binary_at_k(y_scores_sorted, k)
        return metrics.accuracy_score(y_true_sorted, preds_at_k)

    def precision_at_k(self, y_true, y_scores, k):
        '''
        Precision at k (%) population labeled as 1
        '''
        # sort rows by scores
        y_scores_sorted, y_true_sorted = joint_sort_descending(np.array(y_scores), np.array(y_true))
        # label predictions at k 
        preds_at_k = self.generate_binary_at_k(y_scores_sorted, k)
        return metrics.precision_score(y_true_sorted, preds_at_k)

    def recall_at_k(self, y_true, y_scores, k):
        '''
        Recall at k (%) population labeled as 1
        '''
        # sort rows by scores
        y_scores_sorted, y_true_sorted = joint_sort_descending(np.array(y_scores), np.array(y_true))
        # label predictions at k 
        preds_at_k = self.generate_binary_at_k(y_scores_sorted, k)
        return metrics.recall_score(y_true_sorted, preds_at_k)

    def accuracy_at_threshold(self, y_true, y_scores, threshold):
        '''
        Accuracy at probability threshold.
        '''
        y_pred = np.asarray([1 if i >= threshold else 0 for i in y_scores])
        return metrics.accuracy_score(y_true, y_pred)

    def precision_at_threshold(self, y_true, y_scores, threshold):
        '''
        Precision at probability threshold.
        '''
        y_pred = np.asarray([1 if i >= threshold else 0 for i in y_scores])
        return metrics.precision_score(y_true, y_pred)

    def recall_at_threshold(self, y_true, y_scores, threshold):
        '''
        Recall at probability threshold.
        '''
        y_pred = np.asarray([1 if i >= threshold else 0 for i in y_scores])
        return metrics.recall_score(y_true, y_pred)

    def generate_binary_at_k(self, y_scores_sorted, k):
        '''
        Generate binary prediction labels so that top k (%) highest scores are labeled as 1

        Inputs: 
            y_scores_sorted: (list) sorted (desc) prediction scores
            k: (float between 0 and 1) percentage of population labeled as 1
        Returns:
            (array) binary labels
        '''
        cutoff_index = int(len(y_scores_sorted) * k)
        test_predictions_binary = [1 if x < cutoff_index else 0 for x in range(len(y_scores_sorted))]
        return test_predictions_binary

    ######## Plot ########
    def plot_roc(self, save_output=True):
        '''
        Plot ROC curve for the Model.

        Inputs:
            name: (str) plot title
            y_scores: (list) predicted scores
            y_true: (list) y_true from test set
            save_output: (str) optional, True to save fig or False to show fig only.  
                         Default=True
        '''
        # get false postive rates and true positive rates

        fpr, tpr, _ = roc_curve(self.y_test, self.y_scores)

        # get AUC score
        if not self.roc_auc:
            self.roc_auc = auc(fpr, tpr)

        # plot roc curve
        plt.clf()
        pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % self.roc_auc)
        pl.plot([0, 1], [0, 1], 'k--')
        pl.xlim([0.0, 1.05])
        pl.ylim([0.0, 1.05])
        pl.xlabel('False Positive Rate')
        pl.ylabel('True Positive Rate')
        title = "ROC curve: {}-{} {}".format(self.N, self.iteration, self.model_type)
        pl.title(title)
        pl.legend(loc="lower right")

        # save fig if applicable
        if save_output:
            filename = "roc_{}-{}_{}".format(self.N, self.iteration,self.model_type)
            plt.savefig(os.path.join(self.output_dir, filename))
        else:
            plt.show()

    def plot_precision_recall_curve(self, save_output=True):
        '''
        Plot precision recall at n

        Inputs:
            model_name: (str)
            y_scores: (list) predicted scores
            y_true: (list) y_true from test set
            save_output: (str) optional, True to save fig or False to show fig only.  
                         Default=True
        '''
        # get precision curve, recall curve, and the corresponding probability thresholds 
        precision_curve, recall_curve, pr_thresholds = precision_recall_curve(self.y_test, self.y_scores)
        precision_curve = precision_curve[:-1]
        recall_curve = recall_curve[:-1]

        # transfrom probability thresholds into population percentage thresholds
        pct_above_pr_thresholds = []
        num_total = len(self.y_scores)
        for value in pr_thresholds:
            num_above = len(self.y_scores[self.y_scores>=value])
            pct_above = num_above / num_total
            pct_above_pr_thresholds.append(num_above / num_total)
        pct_above_pr_thresholds = np.array(pct_above_pr_thresholds)
        
        # plot curves
        plt.clf()
        fig, ax1 = plt.subplots()
        ax1.plot(pct_above_pr_thresholds, precision_curve, 'b')
        ax1.set_xlabel('percent of population')
        ax1.set_ylabel('precision', color='b')
        ax2 = ax1.twinx()
        ax2.plot(pct_above_pr_thresholds, recall_curve, 'r')
        ax2.set_ylabel('recall', color='r')
        ax1.set_ylim([0,1])
        ax1.set_ylim([0,1])
        ax2.set_xlim([0,1])
        title = "Precision Recall at k: {}-{} {}".format(self.N, self.iteration, self.model_type)
        plt.title(title)

        # save fig if applicable
        if save_output:
            filename = "precision-recall_{}-{}_{}".format(self.N, self.iteration,self.model_type)
            plt.savefig(os.path.join(self.output_dir, filename))
        else:
            plt.show()

#### Helper functions

def joint_sort_descending(l1, l2):
    '''
    Joint sort descending by l1.

    Inputs: l1, l2: numpy arrays
    Returns: tuple of two sorted arrays
    '''
    idx = np.argsort(l1)[::-1]
    return l1[idx], l2[idx]
