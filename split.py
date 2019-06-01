import os
import geopandas as gpd
import pandas as pd
import scipy
from shapely.geometry import shape
from dateutil import parser
from datetime import timedelta
import pickle
import missingno as msno
import matplotlib.pyplot as plt
import pipeline as pp
pd.options.display.max_columns = 999

SDT = 'license_start_date'
EDT = 'expiration_date'
IDT = 'date_issued'
START = parser.parse('1995-02-16') # earliest lic start date in original data
END = parser.parse('2020-07-16') # latest lic start date in original data
PREDICTOR_COLS = ['application_type',
       'conditional_approval', 'date_issued', 
       'expiration_date', 'latitude', 
       'license_code', 
       'license_start_date', 'license_status',
       'license_status_change_date', 'location', 'longitude',
       'police_district', 'precinct', 'site_number', 'ssa', 'ward',
       'ward_precinct', 'zip_code', 'business_id', 'geometry',
       'RegionID', 'census_tract']
LABEL = 'closed'

### Train test split

def prepare(lcs):
    pipeline = pp.Pipeline()
    pipeline.load_clean_data(lcs)
    pipeline.get_train_test_times(START, END, 2, 2)

    for i, (train_start, train_end, test_start, test_end) in enumerate(pipeline.train_test_times):
        print('N = {}'.format(i))
        print('TRAIN: START {} END {}'.format(train_start, train_end))
        print('TEST: START {} END {}\n'.format(test_start, test_end))

    return pipeline


### train test split for one set of times

def train_test_split(lcs, time_col, train_start, train_end, test_start, test_end,
                   predictor_cols=PREDICTOR_COLS, label=LABEL):
    train_df = lcs[(lcs[time_col] >= train_start) & (lcs[time_col] <= train_end)]
    test_df = lcs[(lcs[time_col] >= test_start) & (lcs[time_col] <= test_end)]
    lag_df = lcs[(lcs[time_col] > train_end) & (lcs[time_col] < test_start)]

    train_latest = get_latest_license(train_df)
    test_latest = get_latest_license(test_df)
    
    generate_label(train_latest, train_end, lag_df)
    generate_label(test_latest, test_end, lag_df)

    X_train, y_train, X_test, y_test = split_Xy(train_latest, test_latest, predictor_cols, label)

    return X_train, y_train, X_test, y_test


# ## Before generating outcome: get latest license and other info
# 
# For each business:
# 1. Identify if business has applied for special type licenses in the 2-year period
# 2. Get the total number of licenses applied 
# 3. Get the latest license (latest start date)


def get_special_apptypes_by_business(df):
    '''get dummies for whether business has applied for non-issuance/renewal licenses
    within the period'''
    app_type_dummies = pd.get_dummies(df['application_type'], prefix='applied_for_type')
    other_app_type_dummies = app_type_dummies[['applied_for_type_C_CAPA', 
                                               'applied_for_type_C_EXPA',
                                               'applied_for_type_C_LOC']]
    temp = pd.concat([df, other_app_type_dummies], axis=1)
    other_type_by_bus = temp.groupby('business_id').agg(
                                    {'applied_for_type_C_CAPA': 'sum', 
                                     'applied_for_type_C_EXPA': 'sum',
                                     'applied_for_type_C_LOC': 'sum'}).reset_index()
    return other_type_by_bus


def get_num_licenses_applied(df):
    return df.groupby('business_id').size().reset_index().rename(columns={0: 'num_licenses'})


def get_latest_license(df):
    other_type_by_bus = get_special_apptypes_by_business(df)
    num_licenses = get_num_licenses_applied(df)
    latest = df[df['application_type'].isin(['RENEW','ISSUE'])].groupby(
             'business_id').tail(1).merge(other_type_by_bus, on='business_id').merge(
             num_licenses, on='business_id')
    return latest


# ## Generate outcome

def generate_label(latest, set_end, lag_df):
    
    # init
    latest['closed'] = 0
    
    # gen 1s
    for i, row in latest.iterrows():
        # latest license expires before train/test set end date -> closed
        if row[EDT] < set_end:
            latest.loc[i, 'closed'] = 1
        # latest license expires after end date, but not approved -> closed
        elif row['license_status'] != 'AAI':
            latest.loc[i, 'closed'] = 1
        # latest license expires after end date and are approved, but business did
        # not renew during lag period -> closed
        elif row['business_id'] in lag_df['business_id']:
            latest.loc[i, 'closed'] = 1


def split_Xy(train, test, predictor_cols, label):
    X_train = train[predictor_cols]
    y_train = train[label]
    X_test = test[predictor_cols]
    y_test = test[label]
    return X_train, y_train, X_test, y_test
