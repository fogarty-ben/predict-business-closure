import pandas as pd
import seaborn as sns
import warnings
import numpy as np
import json
import scipy as sp
from sodapy import Socrata
import matplotlib.pyplot as plt
from IPython.display import display


# Annexure I: CTA rides data [from API]
def get_rides(token):
    '''
    Preparing cta rides at ZIP and Ward level
    Input: token for chicago data portal
    Output: CTA monhtly and average weekly rides by Month Year for Zip and Ward level

    Sample Code Run: 
    token = xfs.load_tokens('tokens.json')
    rides_zip, rides_ward= xfs.get_rides(token)
    '''
    client = Socrata('data.cityofchicago.org', token['chicago_open_data_portal'])

    # Get CTA data
    c_rides = client.get('t2rn-p8d7', limit=50000)
    cta=pd.DataFrame.from_dict(c_rides)

    # Process & Subset relevant columns CTA
    cta['month_beginning'] = pd.to_datetime(cta['month_beginning'])
    cta['month_year'] = cta['month_beginning'].dt.to_period('M')
    cta_sub = cta[['station_id','month_year','avg_weekday_rides','monthtotal']]
    cta_sub.loc[:,'station_id']=cta_sub['station_id'].astype(str)

    # Get CTA Station Mapping file to avoid another spatial join
    c_map = client.get('zbnc-zirh')
    cta_map=pd.DataFrame.from_dict(c_map)

    # Rename geo_id columns using dictionary (as specified in mapping file) from on CTA station geocodes 
    # the said variable names are defined in the source
    # https://data.cityofchicago.org/Transportation/CTA-System-Information-List-of-L-Stops-Map/zbnc-zirh
    dic_map={':@computed_region_awaf_s7ux':'Historical Wards 2003-2015',
             ':@computed_region_6mkv_f3dw':'Zip Codes',
             ':@computed_region_vrxf_vc4k':'Community Areas',    
             ':@computed_region_bdys_3d7i':'Census Tracts',
             ':@computed_region_43wa_7qmu':'Wards'}
    cta_map=cta_map.rename(columns = dic_map)
    # Flatten records at by Zip Ward and Station Mapping ID
    cta_map=cta_map.groupby(['map_id','Zip Codes','Wards'], as_index=False).size().reset_index(name='freq')
    cta_map=cta_map.drop('freq',axis=1)
    cta_map['map_id']=cta_map['map_id'].astype(str)

    # Get Ward and ZIP codes to rides data using mapping file
    ct = cta_sub.merge(cta_map, left_on='station_id', right_on='map_id', how='inner')

    # Consistent data types
    ct['Zip Codes']=ct['Zip Codes'].astype(str)
    ct['Wards']=ct['Wards'].astype(str)
    ct['monthtotal']=ct['monthtotal'].astype('float')
    ct['avg_weekday_rides']=ct['avg_weekday_rides'].astype('float')

    # Aggregate by Zip and Ward
    cta_zip=ct.groupby(['month_year','Zip Codes'],as_index=False).agg({"monthtotal": 'sum', 'avg_weekday_rides':'sum'})
    cta_ward=ct.groupby(['month_year','Wards'],as_index=False).agg({"monthtotal": 'sum', 'avg_weekday_rides':'sum'})

    return cta_zip, cta_ward


# Annexure II: Zillow Real Estate square feet price data [from downloaded file]
def get_realestate(nbh_filepath, zip_filepath):
    '''
    Preparing real-estate median square feet price at ZIP and Neighborhood level
    Input: Input filepaths for Zillow Data
    Output: Median Square foot value at NBH and ZIP level by month year (zn_l: zillow:nbh:long form)

    Sample Code Run: 
    nbh_filepath = 'data/raw/ZLW_Neighborhood_MedianValuePerSqft_AllHomes.csv'
    zip_filepath = 'data/raw/ZLW_Zip_MedianValuePerSqft_AllHomes.csv'
    sqft_nbh, sqft_ward = xfs.get_realestate(nbh_filepath,zip_filepath)
    '''

    # Load & subset Sqaure foot price data at Neighborhood
    zlm = pd.read_csv(nbh_filepath)
    zlm_ch = zlm.loc[zlm['City']=='Chicago']
    zlm_ch = zlm_ch.drop(['City', 'State', 'Metro','CountyName'], axis=1)

    # Load & subset Sqaure foot price data at Zip
    zmh = pd.read_csv(zip_filepath,encoding='latin-1')
    zmh_ch = zmh.loc[zmh['City']=='Chicago']
    zmh_ch=zmh_ch.drop(['City', 'State', 'Metro','CountyName'], axis=1)

    # Pivot Attributes for Reshaping Wide to Long format for aggregation
    piv=['RegionID','RegionName','SizeRank']
    val=zlm_ch.columns[~zlm_ch.columns.isin(piv)]
    # Neighborhood
    zn_l = zlm_ch.melt(id_vars=piv,  value_vars=val, var_name='Month', value_name='MedianValuePerSqfeet_Nbh')
    # Zip
    zz_l = zmh_ch.melt(id_vars=piv,  value_vars=val, var_name='Month', value_name='MedianValuePerSqfeet_Zip')

    return zn_l, zz_l


# Annexure III: Economy data on GDP and Unemployment
def get_ecofeatures(ump_filepath,gdp_filepath):
    '''
    Preparing micro market recovery program permits by ZIP and ward level
    Input: input filepaths
    Output: unemployment and gdp 
    ump, gdp = xfs.get_ecofeatures(ump_filepath,gdp_filepath)
    '''
    # gdp_filepath = 'data/raw/Chicago_gdp2001-2017.xlsx'
    # ump_filepath = 'data/raw/Chicago_unemp_2001-2018.xlsx' 

    # Load & subset Sqaure foot price data at Neighborhood
    ump = pd.read_excel(ump_filepath, sheet_name='Data')
    gdp = pd.read_excel(gdp_filepath, sheet_name='Data')

    return ump, gdp


# Auxiliary: API data
def load_tokens(tokens_file):
    '''
    Loads dictionary of API tokens.

    tokens_file (str): path to a JSON file containing API tokens
    '''
    with open(tokens_file,'r') as file:
        return json.load(file)
