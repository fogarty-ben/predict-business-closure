'''
Functions for exploring data

Aya Liu

'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
import missingno as msno


def summarize_missing(data):
    '''
    '''
    print("########")
    print("Number and percentage of missing data in each column:\n")

    # get number and % of NA values for each column
    nas = data.isnull().sum().to_frame()
    nas.columns = ['num_NA']
    nas['perc_NA'] = nas['num_NA']/len(data)
    display(nas[nas['num_NA'] != 0])

    # plot NA matrix
    msno.matrix(data)


def explore_cat_vars(label, cat_vars, data):
    '''
    '''
    for x in cat_vars:
        # Distribution of a categorical variable
        data.groupby(x).size().sort_values(ascending=False).plot(
            kind = 'bar')
        print("\n########")
        print("Number of observations by {}".format(x))
        plt.show()

        # Distribution of a categorical variable grouped by label
        grouped = data.groupby([label, x]).size().reset_index()
        sns.barplot(x=label, y=0, hue=x, data=grouped)
        plt.show()


def explore_num_vars(label, num_vars, data):
    '''
    Calculate and visualize distribution of numerical variables

    '''
    
    # Distribution of all numerical variables
    print('Distribution of all numerical variables')
    display(data[num_vars].describe())

    # Distribution of each numerical variables grouped by label:
    for x in num_vars:
        print("\n########")
        print("Variable: {}\n---".format(x))
        print("Distribution of {} grouped by {}".format(x, label))
        stats = summarize_by_label(label, x, data)
        display(stats)
        print("")

        # Visualize distribution
        plot_num_dist_by_label(label, x, data)

        # Calculate upper extreme and lower extreme to isolate outliers
        ue, le = get_extremes(stats)
        # Check percentage of outliers in each label group
        get_outlier_perc(label, x, ue, le, data)


def summarize_by_label(label, var, data):
    '''
    Get summary statistics for a variable grouped by the target variable
    
    Inputs:
        label: (str) target variable to group by
        var: (str) variable to summarize
        data: (dataframe) a dataframe
    Returns: a pandas dataframe containing summary statistics
    
    '''
    return data.groupby(label)[var].describe()


def get_extremes(stats):
    # Calculate upper extreme and lower extreme to isolate outliers

    ue = 0
    le = float('inf')
    for i in range(len(stats)):
        iqr = stats.iloc[i, 6] - stats.iloc[i, 4] # IQR = Q3 - Q1
        ue_i = stats.iloc[i, 6] + iqr * 1.5 # Upper Extreme = Q3 + 1.5 IQR
        le_i = stats.iloc[i, 4] - iqr * 1.5 # Lower Extreme = Q1 - 1.5 IQR
        if ue_i > ue:
            ue = ue_i
        if le_i < le:
            le = le_i
    return ue, le


def plot_num_dist_by_label(label, var, data):
    '''
    Plot distribution of a variable grouped by the target variable.
    One boxplot includes outliers and one excludes outliers.
    
    Inputs:
        label: (str) target variable to group by
        var: (str) variable to plot
        data: (dataframe) a dataframe

    '''
    # Distribution with outliers
    plt.subplot(1, 2, 1)
    sns.boxplot(x=label, y=var, showfliers=True, data=data)
    plt.title('With outliers')

    # Distribution without outliers
    plt.subplot(1, 2, 2)
    sns.boxplot(x=label, y=var, showfliers=False, data=data)
    plt.title('Without outliers')
    plt.tight_layout()
    plt.show()


def get_outlier_perc(label, var, upper_extreme, lower_extreme, data):
    '''
    Shows the percentage of number of outliers among all observations
    (count of outliers divided by count of all observations) for each group
    splitted by the target variable.

    Inputs:
        label: (str) target variable to group by
        var: (str) variable to plot
        upper_extreme: (int, float) upper extreme above which observations are 
                   considered outliers
        lower_extreme: (int, float) upper extreme below which observations are 
                   considered outliers
        data: (dataframe) a dataframe
    
    '''
    print('Upper outliers: % obs having {} > {}'.format(var, upper_extreme))
    for g in list(data.groupby(label).groups):
        text = '\t{} = {}'.format(label, g)
        perc = len(data[data[label] == g][data[var] > upper_extreme]) / \
               len(data[data[label] == g])
        print("{}: {:.2f}\n".format(text, perc))

    print('Lower outliers: % obs having {} < {}'.format(var, lower_extreme))
    for g in list(data.groupby(label).groups):
        text = '\t{} = {}:'.format(label, g)
        perc = len(data[data[label] == g][data[var] < lower_extreme]) / \
               len(data[data[label] == g])
        print("{}: {:.2f}\n".format(text, perc))

def hist_num_var(varname, target, data, bins):
    '''
    generate histograms for a numerical variable
    all rows, target=1, target=0 
    '''
    data.hist(column=varname, bins=bins)
    plt.title("All observations")
    for g in data.groupby(target).groups:
        data[data[target] == g].hist(column=varname, bins=bins)
        plt.title("{} = {}".format(target, g))
    plt.show()

