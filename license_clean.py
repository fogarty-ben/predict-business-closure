import os
import geopandas as gpd
import pandas as pd
import scipy
from shapely.geometry import shape
from dateutil import parser
from datetime import timedelta
import pickle
import json
import shapely
from sodapy import Socrata
import missingno as msno
import matplotlib.pyplot as plt

import data_preprocess as prep
import data_explore as exp
import pipeline as pp

pd.options.display.max_columns = 999

SDT = 'license_start_date'
EDT = 'expiration_date'
IDT = 'date_issued'
ZILLOW_GEO = 'data/ZillowNeighborhoods-IL.shp'


##### REPEATED FROM census_features.py
def load_tokens(tokens_file):
    '''
    Loads dictionary of API tokens.

    tokens_file (str): path to a JSON file containing API tokens
    '''
    with open(tokens_file,'r') as file:
        return json.load(file)
#####


def get_lcs_data(pickle=True):

    lcs = obtain_lcs()
    lcs = convert_lcs_dtypes(lcs)
    lcs = clean_lcs(lcs)

    #### add geographies ####
    lcs = gdf_from_latlong(lcs, lat='latitude', long='longitude')  

    # add zillow neighborhoods
    nbh = gpd.read_file(ZILLOW_GEO)
    nbh = nbh[nbh.City == 'Chicago'].drop(columns=['State', 'County', 'City']) 
    lcs = add_geography_id(lcs, nbh)
    
    # add census tracts
    lcs = add_census_tracts(lcs)

    # sort by start date
    lcs.sort_values(by=SDT, inplace=True)

    # if pickle:
    #     if not os.path.exists('pickle'):
    #         os.mkdir('pickle')
    #     pickle.dump(lcs_raw, open("pickle/lcs", "wb" ))

    return lcs


def obtain_lcs():
    tokens = load_tokens('tokens.json')
    client = Socrata('data.cityofchicago.org', tokens['chicago_open_data_portal'])
    results = client.get('xqx5-8hwx', city='CHICAGO', limit='9999999999')
    lcs = pd.DataFrame.from_records(results)
    return lcs

def convert_lcs_dtypes(lcs):
    lcs_dates = ['application_created_date', 
                 'application_requirements_complete', 
                 'payment_date',
                 'license_start_date',
                 'expiration_date',
                 'license_approved_for_issuance',
                 'date_issued',
                 'license_status_change_date']
    lcs[lcs_dates] = lcs[lcs_dates].astype('datetime64')   
    lcs['latitude'] = lcs['latitude'].astype('float64')
    lcs['longitude'] = lcs['longitude'].astype('float64')
    return lcs

def clean_lcs(lcs):
    '''
    clean business licenses data
    '''

    # add business id: account_number-site_number
    lcs['business_id'] = lcs['account_number'] + "-" + lcs['site_number'].map(str)

    # fill license start dates
    # for issuance type: fill start date with issue date
    nastart_issue = lcs[SDT].isna() & (lcs['application_type'] == 'ISSUE')
    lcs.loc[nastart_issue, SDT] = lcs.loc[nastart_issue, IDT]
    # for other types: drop (negligible)
    lcs = lcs.dropna(subset=[SDT], axis=0)

    # drop rows with negative license length
    lcs = lcs[(lcs[EDT] - lcs[SDT]) > timedelta(days=0)]

    # drop rows with no location
    lcs = lcs.dropna(subset=['location'], axis=0)
    return lcs


def add_census_tracts(lcs):   
    # census tracts
    # post 2010
    tokens = load_tokens('tokens.json')
    client = Socrata('data.cityofchicago.org', tokens['chicago_open_data_portal'])
    
    tracts10 = pd.DataFrame(client.get('74p9-q2aq', select='the_geom,tractce10'))
    tracts10['the_geom'] = tracts10.the_geom\
                                       .apply(shapely.geometry.shape)
    tracts10 = gpd.GeoDataFrame(tracts10, geometry='the_geom')
    lcs_10 = add_geography_id(lcs[lcs.license_start_date >= parser.parse('2010-01-01')],
                              tracts10)
    lcs_10.rename(columns={'tractce10':'census_tract'}, inplace=True) 

    # pre 2010
    tracts00 = pd.DataFrame(client.get('4hp8-2i8z', select='the_geom,census_tra'))
    tracts00['the_geom'] = tracts00.the_geom\
                                       .apply(shapely.geometry.shape)
    tracts00 = gpd.GeoDataFrame(tracts00, geometry='the_geom')
    lcs_00 = add_geography_id(lcs[lcs.license_start_date < parser.parse('2010-01-01')], 
                              tracts00)
    lcs_00.rename(columns={'census_tra':'census_tract'}, inplace=True) 

    # combine
    lcs = pd.concat([lcs_00, lcs_10], axis=0)
    return lcs


def gdf_from_latlong(df, lat, long):
    '''convert a pandas dataframe to a geodataframe on lat long'''
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[long], df[lat]))
    return gdf


def add_geography_id(gdf, b):
    '''
    merge with a geography boundaries file

    Input:
        gdf: (GeoDataFrame) 
        b: (Geojson) boundaries
    Returns: (GeoDataFrame)
    '''
    b.crs = {'init': 'epsg:4326'}
    gdf.crs = {'init': 'epsg:4326'}

    result = gpd.sjoin(gdf, b, how="left")
    # drop extra column
    result.drop('index_right', axis=1, inplace=True)
    return result


# # sort data by license start date
# lcs.sort_values(by=SDT, inplace=True)


# # In[26]:


# lcs.dtypes


# # # Link data
# # 
# # to be done together

# # # Train test split

# # In[27]:


# pipeline = pp.Pipeline()
# pipeline.load_clean_data(lcs)


# # In[28]:


# start = parser.parse('1995-02-16') # earliest lic start date in original data
# end = parser.parse('2020-07-16') # latest lic start date in original data
# pipeline.get_train_test_times(start, end, 2, 2)

# for i, (train_start, train_end, test_start, test_end) in enumerate(pipeline.train_test_times):
#     print('N = {}'.format(i))
#     print('TRAIN: START {} END {}'.format(train_start, train_end))
#     print('TEST: START {} END {}\n'.format(test_start, test_end))


# # # For 1 Split: (example)
# # update split in pipeline

# # In[64]:


# # example
# time_col = SDT
# train_start, train_end, test_start, test_end = pipeline.train_test_times[3]


# # In[65]:


# train_df = lcs[(lcs[time_col] >= train_start) & (lcs[time_col] <= train_end)]
# test_df = lcs[(lcs[time_col] >= test_start) & (lcs[time_col] <= test_end)]
# lag_df = lcs[(lcs[time_col] > train_end) & (lcs[time_col] < test_start)]


# # ## Before generating outcome: get latest license and other info
# # 
# # For each business:
# # 1. Identify if business has applied for special type licenses in the 2-year period
# # 2. Get the total number of licenses applied 
# # 3. Get the latest license (latest start date)

# # In[66]:


# def get_special_apptypes_by_business(df):
#     '''get dummies for whether business has applied for non-issuance/renewal licenses
#     within the period'''
#     app_type_dummies = pd.get_dummies(df['APPLICATION TYPE'], prefix='applied_for_type')
#     other_app_type_dummies = app_type_dummies[['applied_for_type_C_CAPA', 
#                                                'applied_for_type_C_EXPA',
#                                                'applied_for_type_C_LOC']]
#     temp = pd.concat([df, other_app_type_dummies], axis=1)
#     other_type_by_bus = temp.groupby('business_id').agg(
#                                     {'applied_for_type_C_CAPA': 'sum', 
#                                      'applied_for_type_C_EXPA': 'sum',
#                                      'applied_for_type_C_LOC': 'sum'}).reset_index()
#     return other_type_by_bus


# # In[67]:


# def get_num_licenses_applied(df):
#     return df.groupby('business_id').size().reset_index().rename(columns={0: 'num_licenses'})


# # In[68]:


# def get_latest_license(df):
#     other_type_by_bus = get_special_apptypes_by_business(df)
#     num_licenses = get_num_licenses_applied(df)
#     latest = df[df['APPLICATION TYPE'].isin(['RENEW','ISSUE'])]                .groupby('business_id')                .tail(1)                .merge(other_type_by_bus, on='business_id')                .merge(num_licenses, on='business_id')
#     return latest


# # In[69]:


# # get latest renew/issue license for each business
# train_latest = get_latest_license(train_df)
# test_latest = get_latest_license(test_df)


# # ## Generate outcome

# # In[75]:


# def generate_label(latest, set_end, lag_df):
    
#     # init
#     latest['closed'] = 0
    
#     # gen 1s
#     for i, row in latest.iterrows():
#         # latest license expires before train/test set end date -> closed
#         if row[EDT] < set_end:
#             latest.loc[i, 'closed'] = 1
#         # latest license expires after end date, but not approved -> closed
#         elif row['LICENSE STATUS'] != 'AAI':
#             latest.loc[i, 'closed'] = 1
#         # latest license expires after end date and are approved, but business did
#         # not renew during lag period -> closed
#         elif row['business_id'] in lag_df['business_id']:
#             latest.loc[i, 'closed'] = 1


# # In[76]:


# # generate label for train
# generate_label(train_latest, train_end, lag_df)


# # In[78]:


# # generate label for test
# generate_label(test_latest, test_end, lag_df)


# # In[77]:


# train_latest.groupby('closed').size()


# # In[79]:


# test_latest.groupby('closed').size()


# # ## Get X_train, y_train, X_test, y_test

# # In[508]:


# def split_Xy(train, test, predictor_cols, label):
#     X_train = train[predictor_cols]
#     y_train = train[label]
#     X_test = test[predictor_cols]
#     y_test = test[label]
#     return X_train, y_train, X_test, y_test


# # In[575]:


# label = 'closed'
# predictor_cols = ['ZIP CODE', 'WARD', 'PRECINCT', 'WARD PRECINCT', 
#                   'POLICE DISTRICT', 'LICENSE CODE',
#                   'APPLICATION TYPE', 'APPLICATION CREATED DATE', 
#                   'APPLICATION REQUIREMENTS COMPLETE', 
#                   'PAYMENT DATE', 'CONDITIONAL APPROVAL', 
#                   'LICENSE TERM START DATE', 'LICENSE TERM EXPIRATION DATE', 
#                   'LICENSE APPROVED FOR ISSUANCE', 'DATE ISSUED', 
#                   'LICENSE STATUS', 'LICENSE STATUS CHANGE DATE', 
#                   'SSA', 'LATITUDE', 'LONGITUDE', 'geometry', 'applied_for_type_C_CAPA', 
#                   'applied_for_type_C_EXPA', 'applied_for_type_C_LOC', 'num_licenses']


# # In[576]:


# X_train, y_train, X_test, y_test = split_Xy(train_latest, test_latest, predictor_cols, label)


# # # Feature Engineering

# # ### 1. Application type
# # no change

# # ### 2. License Type

# # In[577]:


# # kept license types w/ > 1% counts
# all_license_codes = X_train['LICENSE CODE'].value_counts(normalize=True).to_frame()
# top_license_codes = all_license_codes[all_license_codes['LICENSE CODE'] > 0.01].index
# prep.keep_top_values(lcs, 'LICENSE CODE', top_license_codes)


# # In[578]:


# top_license_codes


# # In[579]:


# lcs_raw[lcs_raw['LICENSE CODE'].isin(top_license_codes)].groupby(['LICENSE CODE', 'LICENSE DESCRIPTION']).size()


# # ### 3. Number of Open Business in Locale (count of bus w/in two year periods w/in geography)

# # In[584]:


# X_train['ZIP CODE'].value_counts()


# # In[528]:


# lcs_raw.groupby('ZIP CODE').size()


# # In[525]:


# X_train.groupby('ZIP CODE').size()

