import os
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import shape
from dateutil import parser
from dateutil.relativedelta import relativedelta
from datetime import timedelta
import pickle
import json
import shapely
from sodapy import Socrata
import matplotlib.pyplot as plt

pd.options.display.max_columns = 999

SDT = 'license_start_date'
EDT = 'expiration_date'
IDT = 'date_issued'
ZILLOW_GEO = 'data/ZillowNeighborhoods-IL.shp'


def get_lcs_data():

    lcs = obtain_lcs()
    lcs = convert_lcs_dtypes(lcs)
    lcs = clean_lcs(lcs)
    lcs = create_time_buckets(lcs) #need to deal with missing start date before here (apprx. 9863)

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

def create_time_buckets(lcs, bucket_size, start_date=None):
    '''
    Labels each license with a time period. Time periods are defined by the
    bucket size and start date arguments and cut based on the
    license_start_date column.

    Inputs:
    lcs (pandas dataframe): a license dataset
    bucket_size (dictionary): defines the size of each bucket, valid key-value
        pairs are parameters for a dateutil.relativedelta.relativedelta object
    start_date (str): first day to include in a bucket, string of the form
        YYYY-MM-DD or YYYYMMDD
    '''
    if not start_date:
        start_date = min(lcs.license_start_date)

    start_date = pd.to_datetime(start_date)
    bucket_size = relativedelta(**bucket_size)
    lcs['time_period'] = float('nan')


    i = 0
    stop_date = max(lcs.license_start_date)
    while  start_date + i * bucket_size < stop_date:
        start_mask = start_date + i * bucket_size <= lcs.license_start_date
        end_mask = lcs.license_start_date < start_date + (i + 1) * bucket_size
        lcs.loc[start_mask & end_mask, 'time_period'] = i
        i += 1

    return lcs

def create_account_site_time(groupdf):
    '''
    Converts a dataframe of observations with the same account id, site id, and
    time period into a single representative series.

    Inputs:
    groupdf (pandas dataframe): a licenses dataframe where all rows have the
        same account id, site id, and time period.

    Returns: pandas series
    '''
    copy_vals = ['account_number', 'site_number', 'legal_name', 'address',
                 'city', 'state', 'zip_code', 'latitude', 'longitude', 
                 'location', 'police_district', 'precinct', 'ward',
                 'ward_precinct', 'ssa']
    record = groupdf.iloc[0].loc[copy_vals].to_list()

    
    record.append(len(groupdf))

    record.append(min(groupdf.license_start_date))
    record.append(max(groupdf.expiration_date))
    record.append(groupdf.license_code.unique())
    
    record.append(np.mean(groupdf.conditional_approval == 'Y'))
    record.append(np.mean((groupdf.license_status == 'REV') |
                                        (groupdf.license_status == 'REA'))  )  
    record.append(np.mean((groupdf.license_status == 'AAC') |
                                          (groupdf.license_status == 'AAC')))
    record.append(np.mean(groupdf.application_type == 'RENEW'))
    
    return record

def collapse_licenses(lcs):
    '''
    Collapses all the licenses associated with a given accountid-siteid based on
    their time period so that each row represents one accountid-siteid-time
    period.

    Inputs:
    lcs (pandas dataframe): a license dataset

    Returns: pandas dataframe
    '''
    lcs['rev_or_rea'] = (lcs.license_status == 'REV') | (lcs.license_status == 'REA')
    lcs['canceled'] = lcs.license_status == 'AAC'
    lcs.conditional_approval = lcs.conditional_approval == 'Y'
    lcs_collapse = lcs.groupby(['account_number', 'site_number', 'time_period'])\
                      .agg({"license_id": 'count',
                            "legal_name": "first",
                            "doing_business_as_name": 'first',
                            "license_start_date": "min",
                            "expiration_date": "max",
                            "application_type": set,
                            "license_code": set,
                            "license_description": set,
                            "business_activity": set,
                            "business_activity_id": set,
                            "rev_or_rea": 'mean',
                            "canceled": 'mean',
                            "conditional_approval": 'mean',
                            "address": "first",
                            "city": "first",
                            "state": "first",
                            "zip_code": "first",
                            "latitude": "first",
                            "longitude": "first",
                            "location": "first",
                            "police_district": "first",
                            "precinct": "first",
                            "ward": "first",
                            "ward_precinct": "first",
                            "ssa": "first"})\
                      .rename({'license_id': 'n_licenses',
                               'license_start_date': 'min_start_date',
                               'expiration_date': 'max_expiration_date',
                               'application_type': 'application_types',
                               'license_code': 'license_codes',
                               'license_description': 'license_descriptions',
                               'business_activity': 'business_activities',
                               'business_activity_id': 'business_activity_ids',
                               'rev_or_rea': 'pct_revoked',
                               'canceled': 'pct_canceled',
                               'conditional_approval': 'pct_cndtl_approval'},
                              axis=1)

    return lcs_collapse

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


def load_tokens(tokens_file):
    '''
    Loads dictionary of API tokens.

    tokens_file (str): path to a JSON file containing API tokens
    '''
    with open(tokens_file,'r') as file:
        return json.load(file)
