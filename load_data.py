import os
import json
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape
from dateutil import parser
from dateutil.relativedelta import relativedelta
from datetime import timedelta, datetime
import shapely
import certifi
from sodapy import Socrata
import urllib3
import matplotlib.pyplot as plt
import seaborn as sns
pd.options.display.max_columns = 999

MAX_REQS = 1000000
TOKENS_FILEPATH = 'tokens.json'
ZILLOW_FILEPATH = 'data/ZLW_Zip_MedianValuePerSqft_AllHomes.csv'
UMP_FILEPATH = 'data/Chicago_unemp_2001-2018.xlsx'
GDP_FILEPATH = 'data/Chicago_gdp2001-2017.xlsx'
CTA_MONTHS = 6
REAL_ESTATE_MONTHS = 6


#### Main Flow ####
def get_lcs_data(tokens_filepath=TOKENS_FILEPATH, cta_months=CTA_MONTHS, 
                 real_estate_months=REAL_ESTATE_MONTHS, pickle_filepath=None):
    '''
    Load and transform the business license data for the machine learning pipeline. 
    Integrate data with auxiliary datasets.
    
    Input:
        tokens_filepath (str): filepath to a json file containing API keys. The file should 
            contain two keys: "chicago_open_data_portal" and "us_census_bureau". 
            Default="tokens.json"
        cta_months (int): number of months to aggregate cta ridership data over. Default=6.
        real_estate_months (int): number of months to aggregate zillow median home value over.
            Default=6.
        pickle_filepath (str): pickle filepath for saving the linked dataset. If None, the data
            will not be pickled. Default=None.

    Returns: (geodataframe) a pandas dataframe
    '''
    tokens = load_tokens(tokens_filepath)

    print('Downloading licenses from Chicago Open Data Portal...')
    lcs = obtain_lcs(tokens)

    print('Cleaning data...')
    lcs = convert_lcs_dtypes(lcs)
    lcs = clean_lcs(lcs)

    print('Changing unit of analysis...')
    lcs = change_unit_analysis(lcs, {'years': 1}, '2002-01-01' )

    #### add geographies ####
    print('Creating geospatial properties...')
    lcs = gdf_from_latlong(lcs, lat='latitude', long_='longitude')  
    
    # add census tracts
    print('Linking census tracts...')
    lcs = add_census_tracts(lcs)

    # getting additional datasets
    print('Loading additional datasets...')
    try:
        census_2000 = get_2000_census_data(tokens)
    except json.JSONDecodeError:
        print('---Error obtaining 2000 Census data, retrying request')
        census_2000 = get_2000_census_data(tokens)
    try:
        census_2010 = get_2010_census_data(tokens) 
    except:
        print('---Error obtaining 2010 Census data, retrying request')
        census_2010 = get_2010_census_data(tokens)
    try:
        zbp = get_zbp_data(tokens)
    except json.JSONDecodeError:
        print('---Error obtaining ZBP Census data, retrying request')
        zbp = get_zbp_data(tokens)
    cta_ward = get_rides(tokens)
    real_estate = get_realestate(ZILLOW_FILEPATH)
    ump, gdp = get_ecofeatures(UMP_FILEPATH, GDP_FILEPATH)

    print('Linking additional datasets...')
    lcs = link_zbp_licenses(zbp, lcs)
    lcs = link_census_licenses(census_2000, census_2010, lcs)
    lcs = link_cta_licenses(cta_ward, lcs, cta_months)
    lcs = link_real_estate_licenses(real_estate, lcs, real_estate_months)
    lcs = link_gdp_licenses(gdp, lcs)
    lcs = link_ump_licenses(ump, lcs)

    print('Dropping extraneous columns...')
    drop_cols = ['legal_name', 'doing_business_as_name',
                 'license_descriptions', 'business_activities',
                 'business_activity_ids', 'address', 'city', 'state',
                 'ward_precinct', 'geometry']
    lcs = lcs.drop(drop_cols, axis=1)

    if pickle_filepath != None:
        pickle.dump(lcs, open(pickle_filepath, "wb" ))

    return lcs


#### Loading Functions ####

# Load Chicago business licenses data
def obtain_lcs(tokens):
    '''
    Obtain business licenses data from chicago open data portal.

    Input: 
        tokens (dict): dictionary containing API key for Chicago Open Data Portal
    Returns:
        a pandas dataframe
    '''
    client = Socrata('data.cityofchicago.org', tokens['chicago_open_data_portal'])
    results = client.get('xqx5-8hwx', city='CHICAGO', limit=MAX_REQS)
    lcs = pd.DataFrame.from_records(results)
    return lcs

def change_unit_analysis(lcs, bucket_size, start_date=None, stop_date=None):
    '''
    Labels each license with a time period. Time periods are defined by the
    bucket size and start date arguments and cut based on the date_col
    argument.

    Inputs:
    lcs (pandas dataframe): a license dataset
    bucket_size (dictionary): defines the size of each bucket, valid key-value
        pairs are parameters for a dateutil.relativedelta.relativedelta object
    start_date (str): first day to include in a bucket, string of the form
        YYYY-MM-DD or YYYYMMDD
    stop_date (str): the last prediction day to include in a bucket, string of the form
        YYYY-MM-DD or YYYYMMDD

    Returns: tuple of pandas dataframe, list of bucket starting dates

    Stray setting with copy warning here
    '''
    if not start_date:
        start_date = min(lcs.license_start_date)
    if not stop_date:
        stop_date = max(lcs.expiration_date)


    start_date = pd.to_datetime(start_date)
    stop_date = pd.to_datetime(stop_date)
    bucket_size = relativedelta(**bucket_size)
    
    transformed_df = pd.DataFrame()
    lcs['min_start_date'] = lcs.groupby(['account_number', 'site_number'])\
                               .license_start_date\
                               .transform(min)
    lcs['max_exp_date'] = lcs.groupby(['account_number', 'site_number'])\
                               .expiration_date\
                               .transform(max)

    i = 0
    while  start_date + i * bucket_size <= stop_date:
        pred_date = start_date + i * bucket_size 
        start_mask = lcs.min_start_date < pred_date
        end_mask = pred_date < lcs.max_exp_date
        future_mask = lcs.license_start_date < pred_date
        modified_mask = lcs.license_status_change_date < pred_date
        mask = start_mask & end_mask & future_mask & (modified_mask | \
                                                      lcs.license_status_change_date.isna())
        eligible_lcs = lcs.loc[mask, :].copy()
        eligible_lcs['pred_date'] = pred_date
        eligible_lcs = collapse_licenses(eligible_lcs, bucket_size).reset_index()
        transformed_df = transformed_df.append(eligible_lcs, ignore_index=True, sort=True)        
        i += 1

    transformed_df['no_renew_nextpd'] = transformed_df.max_exp_date < transformed_df.pred_date.apply(
                                        lambda x: x + relativedelta(years=2))

    return transformed_df

def collapse_licenses(lcs, time_gap):
    '''
    Collapses all the licenses associated with a given accountid-siteid based on
    their time period so that each row represents one accountid-siteid-time
    period.

    Inputs:
    lcs (pandas dataframe): a license dataset
    time_gap (dictionary): defines the size of each bucket, valid key-value
        pairs are parameters for a dateutil.relativedelta.relativedelta object

    Returns: pandas dataframe
    '''
    lcs.loc[:, 'rev_or_rea'] = (lcs.license_status == 'REV') | (lcs.license_status == 'REA')
    lcs.loc[:, 'canceled'] = lcs.license_status == 'AAC'
    lcs.loc[:, 'conditional_tf'] = lcs.conditional_approval == 'Y'
    lcs_collapse = lcs.groupby(['account_number', 'site_number'])\
                      .agg({"license_id": 'count',
                            "legal_name": "first",
                            "doing_business_as_name": 'first',
                            "license_start_date": ["min", "max"],
                            "application_type": set,
                            "license_code": set,
                            "license_description": set,
                            "business_activity": set,
                            "business_activity_id": set,
                            "rev_or_rea": 'mean',
                            "canceled": 'mean',
                            "conditional_tf": 'mean',
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
                            "ssa": "first",
                            "pred_date": "first",
                            "max_exp_date": "first"})
    multi_index = lcs_collapse.columns.to_list()
    ind = pd.Index(["_".join(entry) for entry in multi_index])
    lcs_collapse.columns = ind
    lcs_collapse = lcs_collapse.rename({"license_id_count": 'n_licenses',
                                        "legal_name_first": 'legal_name',
                                        "doing_business_as_name_first": 'doing_business_as_name',
                                        'license_start_date_min': 'oldest_lcs_start',
                                        'license_start_date_max': 'newest_lcs_start',
                                        'application_type_set': 'application_types',
                                        "license_code_set": "license_codes",
                                        "license_description_set": "license_descriptions",
                                        "business_activity_set": "business_activities",
                                        "business_activity_id_set": "business_activity_ids",
                                        "rev_or_rea_mean": 'pct_revoked',
                                        "canceled_mean": 'pct_canceled',
                                        "conditional_tf_mean": 'pct_cndtl_approval',
                                        "address_first": "address",
                                        "city_first": "city",
                                        "state_first": "state",
                                        "zip_code_first": "zip_code",
                                        "latitude_first": "latitude",
                                        "longitude_first": "longitude",
                                        "location_first": "location",
                                        "police_district_first": "police_district",
                                        "precinct_first": "precinct",
                                        "ward_first": "ward",
                                        "ward_precinct_first": "ward_precinct",
                                        "ssa_first": "ssa",
                                        "pred_date_first": "pred_date",
                                        "max_exp_date_first": "max_exp_date"},
                                        axis=1)

    return lcs_collapse

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

def clean_lcs(lcs): #May want to talk about what we're doing in this function
    '''
    Clean business licenses data.

    Input: 
        lcs: (dataframe) raw business licenses data
    Returns: updated dataframe    
    '''
    # fill license start dates
    # for issuance type: fill start date with issue date
    nastart_issue = lcs['license_start_date'].isna() & (lcs['application_type'] == 'ISSUE')
    lcs.loc[nastart_issue, 'license_start_date'] = lcs.loc[nastart_issue, 'date_issued']
    # for other types: drop (negligible)
    lcs = lcs.dropna(subset=['license_start_date'], axis=0)

    # drop rows with negative license length
    lcs = lcs[(lcs['expiration_date'] - lcs['license_start_date']) > timedelta(days=0)]

    # drop rows with no location
    lcs = lcs.dropna(subset=['location'], axis=0)
    return lcs

def gdf_from_latlong(df, lat, long_):
    '''
    convert a pandas dataframe to a geodataframe on lat long

    Inputs:
        df: (dataframe) original df
        lat: (str) column name for latitude
        long: (str) column name for longitude
    
    Returns: a geodataframe
    '''
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[long_], df[lat]))

    return gdf

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
    
    tracts10 = pd.DataFrame(client.get('74p9-q2aq', select='the_geom,tractce10',
                            limit=MAX_REQS))
    tracts10['the_geom'] = tracts10.the_geom\
                                       .apply(shapely.geometry.shape)
    tracts10 = gpd.GeoDataFrame(tracts10, geometry='the_geom')
    lcs_10 = add_geography_id(lcs[lcs.pred_date >= parser.parse('2010-01-01')],
                              tracts10)
    lcs_10.rename(columns={'tractce10':'census_tract'}, inplace=True) 

    # pre 2010
    tracts00 = pd.DataFrame(client.get('4hp8-2i8z', select='the_geom,census_tra',
                            limit=MAX_REQS))
    tracts00['the_geom'] = tracts00.the_geom\
                                       .apply(shapely.geometry.shape)
    tracts00 = gpd.GeoDataFrame(tracts00, geometry='the_geom')
    lcs_00 = add_geography_id(lcs[lcs.pred_date < parser.parse('2010-01-01')], 
                              tracts00)
    lcs_00.rename(columns={'census_tra':'census_tract'}, inplace=True) 

    # combine
    lcs = pd.concat([lcs_00, lcs_10], axis=0)
    return lcs

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


# Load Census data
def get_census_data(base_url, vars, for_level, in_levels=None, key=None):
    '''
    Downloads a dataset from the United States Census Bureau.

    Inputs:
    base_url (str): the base url of the dataset's API
    vars (list of strs): names of variables to get from the dataset
    for_levels (tuple of (str, list of strs)): the first string denotes the
        geographic level each row data set will represent, and the list of strs
        in the second postion filters the returned results on the geography; to
        avoid filtering, the second entry should be ['*']
    in_levels (dict): geographies used to limit the number of rows in the
        returned dataset; key should be the string name of a geography
        and the value should be the string value of a geography
    key (str): Census Bureau API key
    '''
    http = urllib3.PoolManager(cert_reqs='CERT_REQUIRED', 
                               ca_certs=certifi.where())
    fields = {}

    get_arg = ','.join(vars)
    fields['get'] = get_arg

    for_arg = for_level[0] + ':' + ','.join(for_level[1])
    fields['for'] = for_arg
    
    if in_levels is not None:
        in_args = []
        for geo, val in in_levels.items():
            in_args.append("{}:{}".format(geo, val))
        in_arg = '%20'.join(in_args)
        fields['in'] = in_arg

    if key is not key:
        fields['key'] = key
    
    req = http.request('GET', base_url, fields=fields)
    data = json.loads(req.data)
    df = pd.DataFrame(data[1:], columns=data[0], )
    
    return df

def get_zbp_data(tokens):
    '''
    Temporary function to make merging code easier later.
    '''
    cook_county_code = '031'
    illinois_code = '17'

    zbp_url = 'https://api.census.gov/data/{}/zbp'
    client = Socrata('data.cityofchicago.org', 
                     tokens['chicago_open_data_portal'])
    results = client.get('unjd-c2ca', select='zip')
    chicago_zips = pd.DataFrame.from_records(results).zip.tolist()
    zbp_vars = {'EMP': 'Paid employees for pay period ending March 12',
                'ESTAB': 'Number of establishments',
                'PAYANN': 'Annual payroll'}

    zbp_data = pd.DataFrame()
    for year in range(2000, 2017):
        base_url = zbp_url.format(year)
        zbp_oneyr = get_census_data(base_url, zbp_vars, 
                                    ('zipcode', chicago_zips),
                                    key=tokens['us_census_bureau'])
        zbp_oneyr = zbp_oneyr.rename(zbp_vars, axis=1)
        zbp_oneyr['year'] = year
        zbp_data = zbp_data.append(zbp_oneyr, ignore_index=True)

    zbp_data = zbp_data.astype({'Paid employees for pay period ending March 12': float,
                                'Number of establishments': float,
                                'Annual payroll': float})

    return zbp_data

def process_2000_education(row):
    '''
    Aggregates educations columsn from 2000 decentential census

    row (array like): the row to aggregate over

    returns pandas series
    '''
    processed = pd.Series()
    processed['Less than high school graduate'] = np.sum(row[0:8])
    processed['High school graduate (includes equivalency)'] = row[8]
    processed["Some college or associate's degree"] = np.sum(row[9:12])
    processed["Bachelor's degree"] = row[12]
    processed["Graduate or professional degree"] = np.sum(row[13:16])
    
    if row['P037001']:
        processed = processed / row['P037001'] 
    else:
        processed[:] = float('nan')
    
    return processed

def process_race(row):
    '''
    Processes race columns from 2000 decenentetial census and 2010 ACS

    row (array like): the row to process

    Returns: pandas series
    '''
    processed = pd.Series()
    race_cols = ['White alone', 'Black/AfAmer alone', 'AmInd/Alaskn alone',
                 'Asian alone', 'HI alone', 'Some other race alone',
                 'Total 2+ races']
    if row['Total (Race)']:
        for col in race_cols:
            processed[col] =  row[col] / row['Total (Race)']
    else:
        for col in race_cols:
            processed[col] =  float('nan')

    if row ['Total (Hispanic/Not Hispanic)']:
        processed['Hispanic or Latino'] = (row["Hispanic or Latino"] / 
                                           row["Total (Hispanic/Not Hispanic)"])
    else:
        processed['Hispanic or Latino'] = float('nan')
    
    return processed

def process_2010_education(row):
    '''
    Aggregates educations columsn from 2000 decentential census

    row (array like): the row to aggregate over

    returns pandas series
    '''
    if row['Population 25 years and over']:
        row = row / row['Population 25 years and over']
    else: 
        row[:] = float('nan')
    row = row.drop('Population 25 years and over')
    
    return row

def get_2000_census_data(tokens):
    '''
    Temporary function to make merging code easier later
    '''
    base_urls = {'sf1': 'https://api.census.gov/data/2000/sf1',
                 'sf3': 'https://api.census.gov/data/2000/sf3'}

    #Race
    dec_2000_race = {'P003001': 'Total (Race)',
                    'P003003': 'White alone',
                    'P003004': 'Black/AfAmer alone',
                    'P003005': 'AmInd/Alaskn alone',
                    'P003006': 'Asian alone',
                    'P003007': 'HI alone',
                    'P003008': 'Some other race alone',
                    'P003009': 'Total 2+ races',
                    'P004001': 'Total (Hispanic/Not Hispanic)',
                    'P004002': 'Hispanic or Latino'}

    race_2000 = get_census_data(base_urls['sf1'], dec_2000_race.keys(), 
                                ('tract', ['*']), 
                                in_levels={'state': '17', 'county': '031'},
                                key=tokens['us_census_bureau'])
    
    race_2000 = race_2000.rename(dec_2000_race, axis=1)\
                         .drop(['state', 'county'], axis=1)\
                         .set_index('tract')\
                         .astype(float)\
                         .drop('000000')

    race_2000 = race_2000.apply(process_race, axis=1)

    #Income
    dec_2000_income = {'P053001': 'Median household income (1999 dollars)',
                       'P089001': 'Total: Population for whom poverty status is determined',
                       'P089002': 'Income below poverty level'}
    
    income_2000 = get_census_data(base_urls['sf3'], dec_2000_income.keys(), 
                                  ('tract', ['*']), 
                                  in_levels={'state': '17', 'county': '031'},
                                  key=tokens['us_census_bureau'])
    income_2000 = income_2000.rename(dec_2000_income, axis=1)\
                             .drop(['state', 'county'], axis=1)\
                             .set_index('tract')\
                             .astype(int)\
                             .drop('000000')

    income_2000['Income below poverty level'] = (income_2000['Income below poverty level'] / 
                                                 income_2000['Total: Population for whom poverty status is determined'])
    income_2000 = income_2000.drop('Total: Population for whom poverty status is determined', 
                                   axis=1)

    dec_2000_education =   {"P037001": "Total: Population 25 years and over",
                            "P037003": "Total: Male: No schooling completed",
                            "P037004": "Total: Male: Nursery to 4th grade",
                            "P037005": "Total: Male: 5th and 6th grade",
                            "P037006": "Total: Male: 7th and 8th grade",
                            "P037007": "Total: Male: 9th grade",
                            "P037008": "Total: Male: 10th grade",
                            "P037009": "Total: Male: 11th grade",
                            "P037010": "Total: Male: 12th grade, no diploma",
                            "P037011": "Total: Male: High school graduate (includes equivalency)",
                            "P037012": "Total: Male: Some college, less than 1 year",
                            "P037013": "Total: Male: Some college, 1 or more years, no degree",
                            "P037014": "Total: Male: Associate degree",
                            "P037015": "Total: Male: Bachelor's degree",
                            "P037016": "Total: Male: Master's degree",
                            "P037017": "Total: Male: Professional school degree",
                            "P037018": "Total: Male: Doctorate degree",
                            "P037020": "Total: Female: No schooling completed",
                            "P037021": "Total: Female: Nursery to 4th grade",
                            "P037022": "Total: Female: 5th and 6th grade",
                            "P037023": "Total: Female: 7th and 8th grade",
                            "P037024": "Total: Female: 9th grade",
                            "P037025": "Total: Female: 10th grade",
                            "P037026": "Total: Female: 11th grade",
                            "P037027": "Total: Female: 12th grade, no diploma",
                            "P037028": "Total: Female: High school graduate (includes equivalency)",
                            "P037029": "Total: Female: Some college, less than 1 year",
                            "P037030": "Total: Female: Some college, 1 or more years, no degree",
                            "P037031": "Total: Female: Associate degree",
                            "P037032": "Total: Female: Bachelor's degree",
                            "P037033": "Total: Female: Master's degree",
                            "P037034": "Total: Female: Professional school degree",
                            "P037035": "Total: Female: Doctorate degree"}

    educ_2000 = get_census_data(base_urls['sf3'], dec_2000_education.keys(), 
                                ('tract', ['*']), 
                                in_levels={'state': '17', 'county': '031'},
                                key=tokens['us_census_bureau'])
    
    educ_2000 = educ_2000.set_index('tract')\
                         .astype(int)
    
    educ_2000_combined = pd.DataFrame(data=(educ_2000.iloc[:, 1:17].to_numpy() + 
                                            educ_2000.iloc[:, 17:33].to_numpy()),
                                      index=educ_2000.index)
    educ_2000_combined['P037001'] = educ_2000['P037001'].to_numpy()
    educ_2000 = educ_2000_combined.apply(process_2000_education, axis=1)\
                                  .drop('000000')

    return pd.concat([race_2000, income_2000, educ_2000], axis=1)
    
def get_2010_census_data(tokens):
    '''
    Temporary function to make merging code easier later
    '''
    acs_url = 'https://api.census.gov/data/2010/acs/acs5'

    #ACS 2010 5yr, Detailed Table
    acs_2010_race = {"B02001_001E": "Total (Race)",
                     "B02001_002E": "White alone",
                     "B02001_003E": "Black/AfAmer alone",
                     "B02001_004E": "AmInd/Alaskn alone",
                     "B02001_005E": "Asian alone",
                     "B02001_006E": "HI alone",
                     "B02001_007E": "Some other race alone",
                     "B02001_008E": "Total 2+ races",
                     "B03001_001E": "Total (Hispanic/Not Hispanic)",
                     "B03001_003E": "Hispanic or Latino"}

    race_2010 = get_census_data(acs_url, acs_2010_race.keys(), 
                                ('tract', ['*']),
                                in_levels={'state': '17', 'county': '031'},
                                key=tokens['us_census_bureau'])
    
    race_2010 = race_2010.rename(acs_2010_race, axis=1)\
                         .drop(['state', 'county'], axis=1)\
                         .set_index('tract')\
                         .astype(int)

    race_2010 = race_2010.apply(process_race, axis=1)

    acs_2010_income = {"B19013_001E": "Median household income (1999 dollars)",
                       "B06012_001E": 'Total: Population for whom poverty status is determined',
                       'B06012_002E': 'Income below poverty level'}

    income_2010 = get_census_data(acs_url, acs_2010_income.keys(), 
                                  ('tract', ['*']), 
                                  in_levels={'state': '17', 'county': '031'},
                                  key=tokens['us_census_bureau'])
    
    income_2010 = income_2010.rename(acs_2010_income, axis=1)\
                             .drop(['state', 'county'], axis=1)\
                             .set_index('tract')\
                             .astype(int)

    income_2010['Median household income (1999 dollars)'] = (income_2010['Median household income (1999 dollars)'] * 
                                                             .7640) #adjust for inflation
    income_2010['Income below poverty level'] = (income_2010['Income below poverty level'] /
                                                 income_2010['Total: Population for whom poverty status is determined'])
    income_2010 = income_2010.drop('Total: Population for whom poverty status is determined',
                                   axis=1)

    acs_2010_education =   {"B06009_001E": "Population 25 years and over",
                            "B06009_002E": "Less than high school graduate",
                            "B06009_003E": "High school graduate (includes equivalency)",
                            "B06009_004E": "Some college or associate's degree",
                            "B06009_005E": "Bachelor's degree",
                            "B06009_006E": "Graduate or professional degree"}

    education_2010 = get_census_data(acs_url, acs_2010_education.keys(), 
                                     ('tract', ['*']), 
                                     in_levels={'state': '17', 'county': '031'},
                                     key=tokens['us_census_bureau'])

    education_2010 = education_2010.rename(acs_2010_education, axis=1)\
                                   .drop(['state', 'county'], axis=1)\
                                   .set_index('tract')\
                                   .astype(int)\
                                   .apply(process_2010_education, axis=1)

    return pd.concat([race_2010, income_2010, education_2010], axis=1)


# Load CTA ridership data
def get_rides(tokens):
    '''
    Obtain and aggregate CTA monhtly ridership at ward level.

    Input: 
        tokens (dict): dictionary containing API key for chicago data portal
    Returns: 
        a pandas data frame. 
    '''
    client = Socrata('data.cityofchicago.org', tokens['chicago_open_data_portal'])

    # Get CTA data
    c_rides = client.get('t2rn-p8d7', limit=50000)
    cta = pd.DataFrame.from_dict(c_rides)

    # Process & Subset relevant columns CTA
    cta.loc[:, 'month_beginning'] = pd.to_datetime(cta['month_beginning'])
    cta.loc[:, 'month_year'] = cta['month_beginning'].dt.to_period('M')
    cta_sub = cta.loc[:, ['station_id','month_year','avg_weekday_rides','monthtotal']]
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
    cta_map=cta_map.groupby(['map_id','Wards'], as_index=False).size().reset_index(name='freq')
    cta_map=cta_map.drop('freq',axis=1)
    cta_map.loc[:, 'map_id']=cta_map['map_id'].astype(str)

    # Get Ward and ZIP codes to rides data using mapping file
    ct = cta_sub.merge(cta_map, left_on='station_id', right_on='map_id', how='inner')

    # Convert data types
    ct.loc[:, 'Wards']=ct['Wards'].astype(str)
    ct.loc[:, 'monthtotal']=ct['monthtotal'].astype('float')
    ct.loc[:, 'avg_weekday_rides']=ct['avg_weekday_rides'].astype('float')

    cta_ward=ct.groupby(['month_year','Wards'],as_index=False).agg(
                {"monthtotal": 'sum', 'avg_weekday_rides':'sum'})

    return cta_ward

# Load Zillow median home values
def get_realestate(zip_filepath):
    '''
    Obtain and aggregate real-estate median square feet price at zipcode level
    Input: 
        zip_filepath (str): input filepaths for Zillow Data
    
    Returns: a pandas dataframe
    '''
    # Load & subset Sqaure foot price data at Zip
    zmh = pd.read_csv(zip_filepath,encoding='latin-1')
    zmh_ch = zmh.loc[zmh['City']=='Chicago']
    zmh_ch = zmh_ch.drop(['City', 'State', 'Metro','CountyName'], axis=1)

    # Pivot Attributes for Reshaping Wide to Long format for aggregation
    piv  = ['RegionID','RegionName','SizeRank']
    val = zmh_ch.columns[~zmh_ch.columns.isin(piv)]

    zz_l = zmh_ch.melt(id_vars=piv,  value_vars=val, var_name='Month', 
                       value_name='MedianValuePerSqfeet_Zip')

    return zz_l

# Load macroeconomic data
def get_ecofeatures(ump_filepath, gdp_filepath):
    '''
    Get annual GDP and unemployment rates in Chicago.

    Input: 
        ump_filepath: (str) filepath for umemployment data
        gdp_filepaths: (str) filepath for GDP data
    Returns: tuple of unemployment and GDP data frames (ump, gdp)
    '''
    ump = pd.read_excel(ump_filepath, sheet_name='Data')
    gdp = pd.read_excel(gdp_filepath, sheet_name='Data')

    WINDOW=1
    gdp['GDP_growth'] = gdp['GDP_billion_dollars'].pct_change(periods=WINDOW); 
    gdp = gdp.loc[gdp['Year'] > 2001]

    return ump, gdp

# Helper function for obtaining data
def load_tokens(tokens_file):
    '''
    Loads dictionary of API tokens.

    tokens_file (str): path to a JSON file containing API tokens
    '''
    with open(tokens_file,'r') as file:
        return json.load(file)

#### Linking functions ####
def link_zbp_licenses(zbp, licenses):
    '''
    Links Census data on Zip Code Business Patterns with business licenses based
    on year and zipcodes.

    Inputs:
    zpb (pandas dataframe): a set of zip code bsuiness patterns data
    licenses (pandas dataframe): a set of business licenses data

    Returns: pandas dataframe
    '''
    licenses['year'] = licenses['pred_date'].dt.to_period('Y') - 1
    
    zbp['merge_col'] = zbp.year.astype(str) + '-' + zbp.zipcode.astype(str)
    zbp = zbp.drop('year', axis=1)
    licenses['merge_col'] = licenses.year.astype(str) + '-' + licenses.zip_code.astype(str)
    licenses = licenses.drop('year', axis=1)

    return pd.merge(licenses, zbp, how='left', on='merge_col').drop(['merge_col', 'zipcode'],
                                                                    axis=1)

def link_census_licenses(census_2000, census_2010, licenses):
    '''
    Links tract-level census demographic information with business licenses
    based on year and census tract.

    Inputs:
    census_2000 (pandas dataframe): a set of census demographic data for
        instances between 2000 and 2009
    census_2010 (pandas dataframe): a set of census demographic data for
        instances between 2000 and 2009
    licenses (pandas dataframe): a set of business licenses data

    Returns: pandas dataframe
    '''
    licenses['year'] = licenses['pred_date'].dt.to_period('Y') - 1


    pre_2010_lcs_mask = licenses.year < 2010
    pre_2010 = pd.merge(licenses[pre_2010_lcs_mask], census_2000, how='left',
                        left_on='census_tract', right_on='tract')

    post_2010_lcs_mask = licenses.year > 2010
    post_2010 = pd.merge(licenses[post_2010_lcs_mask], census_2010, how='left',
                         left_on='census_tract', right_on='tract')

    return pre_2010.append(post_2010, ignore_index=True)\
                   .drop('year', axis=1)

def link_cta_licenses(cta, licenses, months):
    '''
    Links 'El' station ridership with business license data based on a given
    time length.

    Inputs:
    cta (pandas dataframe): a set of cta ridership data
    licenses (pandas dataframe): a set of business licenses data
    months (int): number of months to aggregate cta ridership data over

    Returns: pandas dataframe
    '''
    licenses['pred_month_year'] = licenses['pred_date'].dt.to_period('M')

    cta = cta.groupby('Wards')\
             [['month_year', 'monthtotal', 'avg_weekday_rides']]\
             .rolling(window=months, on='month_year')\
             .mean()\
             .droplevel(1)\
             .reset_index()\
             .rename({'monthtotal': 'monthavg_last{}'.format(months),
                      'avg_weekday_rides': 'avg_weekday_rides_last{}'.format(months)},
                     axis=1)
    cta['merge_col'] = cta['Wards'].astype(str) + '_' + cta.month_year.astype(str)

    licenses['merge_col'] = (licenses['ward'].astype(str) + '_' +
                             licenses.pred_month_year.astype(str))

    return pd.merge(licenses, cta, how='left', on='merge_col')\
             .drop(['month_year', 'pred_month_year', 'merge_col', 'Wards'], axis=1)

def link_real_estate_licenses(real_estate, licenses, months):
    '''
    Links median home price per square foot in a given zipcode from Zillow
    with business license data.

    Inputs:
    realestate (pandas dataframe): a set of real estate price data
    licenses (pandas dataframe): a set of business licenses data
    months (int): number of months to aggregate cta ridership data over


    Returns: pandas dataframe
    '''
    licenses['pred_month_year'] = licenses['pred_date'].dt.to_period('M')

    real_estate = real_estate.groupby('RegionName')\
                             [['Month', 'MedianValuePerSqfeet_Zip']]\
                             .rolling(window=months, on='Month')\
                             .mean()\
                             .droplevel(1)\
                             .reset_index()

    real_estate['merge_col'] = real_estate['RegionName'].astype(str) + '_' + real_estate.Month.astype(str)

    licenses['merge_col'] = (licenses['zip_code'].astype(str) + '_' +
                             licenses.pred_month_year.astype(str))

    return pd.merge(licenses, real_estate, how='left', on='merge_col')\
             .drop(['pred_month_year', 'merge_col', 'RegionName', 'Month'], axis=1)

def link_gdp_licenses(gdp, licenses):
    '''
    Links Chicago GDP data with business license data.
    
    Inputs:
    gdp (pandas dataframe): a set of gdp data
    licenses (pandas dataframe): a set of business licenses data
    '''
    licenses['Year'] = licenses['pred_date'].dt.year - 1 

    return pd.merge(licenses, gdp, how='left', on='Year')\
             .drop('Year', axis=1)

def link_ump_licenses(ump, licenses):
    '''
    Links Chicago unemployment rate data with business license data.
    
    Inputs:
    ump (pandas dataframe): a set of gdp data
    licenses (pandas dataframe): a set of business licenses data
    '''
    licenses['Year'] = licenses['pred_date'].dt.year - 1

    return pd.merge(licenses, ump, how='left', on='Year')\
             .drop('Year', axis=1)\
             .rename({'Annual': "unemployment_rate"}, axis=1)


