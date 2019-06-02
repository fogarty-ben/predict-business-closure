import os
import geopandas as gpd
import pandas as pd
from shapely.geometry import shape
from dateutil import parser
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


def get_lcs_data(pickle_data=False):
    '''
    obtain and clean business licenses data 

    returns: (geodataframe) clean business licenses dataframe
    '''
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

    # pickle clean dataframe
    if pickle_data:
        if not os.path.exists('pickle'):
            os.mkdir('pickle')
        pickle.dump(lcs_raw, open("pickle/lcs", "wb" ))

    return lcs


def obtain_lcs():
    '''
    obtain business licenses data from chicago open data portal.
    '''
    tokens = load_tokens('tokens.json')
    client = Socrata('data.cityofchicago.org', tokens['chicago_open_data_portal'])
    results = client.get('xqx5-8hwx', city='CHICAGO', limit='9999999999')
    lcs = pd.DataFrame.from_records(results)
    return lcs

def convert_lcs_dtypes(lcs):
    '''
    Convert data types of business licenses.

    Input: 
        lcs: (dataframe) raw business licenses data
    Returns: updated dataframe
    '''
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
    Clean business licenses data.

    Input: 
        lcs: (dataframe) raw business licenses data
    Returns: updated dataframe    
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
    '''
    add census tract number to business licenses dataset through spatial join
    with census tract boundaries (2000 and 2010) from chicago open data oprtal

    Input: 
        lcs: (geodataframe) business licenses data
    Returns: geodataframe with census tract number
    
    '''
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
    '''
    convert a pandas dataframe to a geodataframe on lat long

    Inputs:
        df: (dataframe) original df
        lat: (str) column name for latitude
        long: (str) column name for longitude
    
    Returns: a geodataframe
    '''
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[long], df[lat]))
    return gdf


def add_geography_id(gdf, b):
    '''
    spatial join gdf with points (lat, long) with another geodataframe with
    polygons (boundaries).

    Input: 
        gdf: geodataframe with points
        b: geodataframe with polygons (boundaries)

    Returns: (geodataframe) gdf with geography id from b
    
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
