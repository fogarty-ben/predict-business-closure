import pandas as pd
import seaborn as sns
import warnings
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from IPython.display import display


# Annexure I: CTA rides data
def get_rides(rides_filepath, map_filepath):
    '''
    Preparing cta rides at zip and ward level
    Input: input filepaths
    Output: none
    '''
    # rides_filepath = 'data/raw/CTA_Ridership_Monthly_Day_Averages_Totals.csv'
    # map_filepath = 'data/CTA_Station_Map.csv'    

    # Load CTA data
    cta = pd.read_csv(rides_filepath, thousands=',')
    cta_sub = cta[['station_id','month_beginning','avg_weekday_rides','monthtotal']]
    cta_sub['station_id']=cta_sub['station_id'].astype(str)

    # Get CTA Mapping file
    cta_map = pd.read_csv(map_filepath)
    cmap_sub = cta_map[['MAP_ID','STOP_NAME','STATION_NAME','Zip Codes','Wards']]
    cmap_sub['MAP_ID']=cmap_sub['MAP_ID'].astype(str)

    # Link using mapping file
    ct = cta_sub.merge(cmap_sub, left_on='station_id', right_on='MAP_ID', how='inner')
    ct['Zip Codes']=ct['Zip Codes'].astype(str)
    ct['Wards']=ct['Wards'].astype(str)

    # Aggregate by Zip and Ward
    cta_zip=ct.groupby(['month_beginning','Zip Codes'],as_index=False).agg({"monthtotal": 'sum', 'avg_weekday_rides':'sum'})
    cta_ward=ct.groupby(['month_beginning','Wards'],as_index=False).agg({"monthtotal": 'sum', 'avg_weekday_rides':'sum'})

    # Write Data if required
    # cta_zip.to_csv(op_zip_filepath,index=False)
    # cta_ward.to_csv(op_ward_filepath,index=False)

    return cta_zip, cta_ward


# Annexure II: Zillow Real Estate square feet price data 
def get_realestate(nbh_filepath, zip_filepath):
    '''
    Preparing real-estate median square feet price at ZIP and Neighborhood level
    Input: input filepaths
    Output: none
    '''
    # nbh_filepath = 'data/raw/ZLW_Neighborhood_MedianValuePerSqft_AllHomes.csv'
    # zip_filepath = 'data/raw/ZLW_Zip_MedianValuePerSqft_AllHomes.csv'

    # Load & subset Sqaure foot price data at Neighborhood
    zlm = pd.read_csv(nbh_filepath)
    zlm_ch = zlm.loc[zlm['City']=='Chicago']
    zlm_ch = zlm_ch.drop(['City', 'State', 'Metro','CountyName'], axis=1)

    # Load & subset Sqaure foot price data at Zip
    zmh = pd.read_csv(zip_filepath,encoding='latin-1')
    zmh_ch = zmh.loc[zmh['City']=='Chicago']
    zmh_ch=zmh_ch.drop(['City', 'State', 'Metro','CountyName'], axis=1)

    # Pivot Attributes for Reshaping
    piv=['RegionID','RegionName','SizeRank']
    val=zlm_ch.columns[~zlm_ch.columns.isin(piv)]
    # Neighborhood
    zn_l = zlm_ch.melt(id_vars=piv,  value_vars=val, var_name='Month', value_name='MedianValuePerSqfeet_Nbh')
    # Zip
    zz_l = zmh_ch.melt(id_vars=piv,  value_vars=val, var_name='Month', value_name='MedianValuePerSqfeet_Zip')

    # Write Data
    # zz_l.to_csv(op_zip_filepath,index=False)
    # zn_l.to_csv(op_nbh_filepath,index=False)

    return zn_l, zz_l


# Annexure III: Micro market recovery program (MMRP) permits data
def get_permits(permit_filepath):
    '''
    Preparing micro market recovery program permits by ZIP and ward level
    Input: input filepaths
    Output: none
    '''
    # permit_filepath = 'data/raw/Micro-Market_Recovery_Program_-_Permits.csv'

    # Load & subset Sqaure foot price data at Neighborhood
    mmr_p=pd.read_csv(permit_filepath)

    # Data type: Date
    mmr_p['Add Date/Time'] = pd.to_datetime(mmr_p['Add Date/Time'])
    mmr_p['month_year'] = mmr_p['Add Date/Time'].dt.to_period('M')

    # Pivot Attributes for Reshaping
    # Zip
    mrp_z=mmr_p.groupby(['month_year','Zip Code']).size().to_frame('MMRPermits').reset_index()
    # Wards
    mrp_w=mmr_p.groupby(['month_year','Ward']).size().to_frame('MMRPermits').reset_index()

    # Write Data
    # mrp_z.to_csv(op_zip_filepath,index=False)
    # mrp_w.to_csv(op_ward_filepath,index=False)

    return mrp_z, mrp_w


# Annexure IV: Economy data on GDP and Unemployment
def get_ecofeatures(ump_filepath,gdp_filepath):
    '''
    Preparing micro market recovery program permits by ZIP and ward level
    Input: input filepaths
    Output: none
    '''
    # gdp_filepath = 'data/raw/Chicago_gdp2001-2017.xlsx'
    # ump_filepath = 'data/raw/Chicago_unemp_2001-2018.xlsx' 

    # Load & subset Sqaure foot price data at Neighborhood
    ump = pd.read_excel(ump_filepath, sheetname='Data')
    gdp = pd.read_excel(gdp_filepath, sheetname='Data')

    # Write Data
    # ump.to_csv(op_ump_filepath,index=False)
    # gdp.to_csv(op_gdp_filepath,index=False)

    return ump, gdp