import json
import numpy as np
import pandas as pd
from sodapy import Socrata
import urllib3

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
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    http = urllib3.PoolManager()
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
    processed = processed / row['P037001'] 
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
    for col in race_cols:
        processed[col] = row[col] / row["Total (Race)"]

    processed['Hispanic or Latino'] = row["Hispanic or Latino"] / row["Total (Hispanic/Not Hispanic)"]
    return processed

def process_2010_education(row):
    '''
    Aggregates educations columsn from 2000 decentential census

    row (array like): the row to aggregate over

    returns pandas series
    '''
    row = row / row['Population 25 years and over']
    row = row.drop('Population 25 years and over')
    return row

def get_zbp_data():
    '''
    Temporary function to make merging code easier later.
    '''
    tokens = load_tokens('tokens.json')

    cook_county_code = '031'
    illinois_code = '17'

    zbp_url = 'https://api.census.gov/data/{}/zbp'
    client = Socrata('data.cityofchicago.org', tokens['chicago_open_data_portal'])
    results = client.get('unjd-c2ca', select='zip')
    chicago_zips = pd.DataFrame.from_records(results).zip.tolist()
    print(len(chicago_zips))
    agg_zip_vars = ['EMP', 'ESTAB', 'PAYANN']

    aggregate = pd.DataFrame()
    for year in range(2000, 2017):
        base_url = zbp_url.format(year)
        aggregate_append = get_census_data(base_url, agg_zip_vars, 
                                           ('zipcode', chicago_zips),
                                           key=tokens['us_census_bureau'])
        aggregate_append['year'] = year
        aggregate = aggregate.append(aggregate_append, ignore_index=True)

    return aggregate

def get_2000_census_data():
    '''
    Temporary function to make merging code easier later
    '''
    tokens = load_tokens('tokens.json')

    base_urls = {'2000_census_sf1': 'https://api.census.gov/data/2000/sf1',
                 '2000_census_sf3': 'https://api.census.gov/data/2000/sf3'}

    #SF1
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

    race_2000 = get_census_data(base_urls['2000_census_sf1'], list(dec_2000_race.keys()), 
                                ('tract', ['*']), in_levels={'state': '17',
                                                             'county': '031'},
                                key=tokens['us_census_bureau'])
    
    race_2000 = race_2000.rename(dec_2000_race, axis=1)\
                         .drop(['state', 'county'], axis=1)\
                         .astype(int)\
                         .set_index('tract')\
                         .drop(000000)

    race_2000 = race_2000.apply(process_race, axis=1)

    #SF3
    dec_2000_income = {'P053001': 'Median household income (1999 Dollars)',
                       'P089001': 'Total: Population for whom poverty status is determined',
                       'P089002': 'Income below poverty level'}
    
    income_2000 = get_census_data(base_urls['2000_census_sf3'], list(dec_2000_income.keys()), 
                                          ('tract', ['*']), in_levels={'state': '17',
                                                                       'county': '031'},
                                          key=tokens['us_census_bureau'])
    income_2000 = income_2000.rename(dec_2000_income, axis=1)\
                             .drop(['state', 'county'], axis=1)\
                             .astype(int)\
                             .set_index('tract')\
                             .drop(000000)

    income_2000['Income below poverty level'] = income_2000['Income below poverty level'] / income_2000['Total: Population for whom poverty status is determined']
    income_2000 = income_2000.drop('Total: Population for whom poverty status is determined', axis=1)

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

    educ_2000 = get_census_data(base_urls['2000_census_sf3'], list(dec_2000_education.keys()), 
                                          ('tract', ['*']), in_levels={'state': '17',
                                                                       'county': '031'},
                                          key=tokens['us_census_bureau'])
    educ_2000 = educ_2000.astype(int)
    educ_2000_combined = pd.DataFrame(data=(educ_2000.iloc[:, 1:17].to_numpy() + educ_2000.iloc[:, 17:33].to_numpy()),
                                      index=educ_2000.tract)
    educ_2000_combined['P037001'] = educ_2000['P037001'].to_numpy()
    educ_2000 = educ_2000_combined.apply(process_2000_education, axis=1)\
                                  .drop(000000)

    return pd.concat([race_2000, income_2000, educ_2000], axis=1)
    
def get_2010_census_data():
    '''
    Temporary function to make merging code easier later
    '''
    tokens = load_tokens('tokens.json')

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

    race_2010 = get_census_data(acs_url, list(acs_2010_race.keys()), 
                                ('tract', ['*']), in_levels={'state': '17',
                                                             'county': '031'},
                                key=tokens['us_census_bureau'])
    
    race_2010 = race_2010.rename(acs_2010_race, axis=1)\
                         .drop(['state', 'county'], axis=1)\
                         .astype(int)\
                         .set_index('tract')

    race_2010 = race_2010.apply(process_race, axis=1)

    acs_2010_income = {"B19013_001E": "Median household income (1999 dollars)",
                       "B06012_001E": 'Total: Population for whom poverty status is determined',
                       'B06012_002E': 'Income below poverty level'}

    income_2010 = get_census_data(acs_url, list(acs_2010_income.keys()), 
                                  ('tract', ['*']), in_levels={'state': '17',
                                                               'county': '031'},
                                  key=tokens['us_census_bureau'])
    income_2010 = income_2010.rename(acs_2010_income, axis=1)\
                             .drop(['state', 'county'], axis=1)\
                             .astype(int)\
                             .set_index('tract')

    income_2010['Median household income (1999 dollars)'] = income_2010['Median household income (1999 dollars)'] * .7640 #adjust for inflation
    income_2010['Income below poverty level'] = income_2010['Income below poverty level'] / income_2010['Total: Population for whom poverty status is determined']
    income_2010 = income_2010.drop('Total: Population for whom poverty status is determined', axis=1)

    acs_2010_education =   {"B06009_001E": "Population 25 years and over",
                            "B06009_002E": "Less than high school graduate",
                            "B06009_003E": "High school graduate (includes equivalency)",
                            "B06009_004E": "Some college or associate's degree",
                            "B06009_005E": "Bachelor's degree",
                            "B06009_006E": "Graduate or professional degree"}

    education_2010 = get_census_data(acs_url, list(acs_2010_education.keys()), 
                                ('tract', ['*']), in_levels={'state': '17',
                                                             'county': '031'},
                                key=tokens['us_census_bureau'])
    

    education_2010 = education_2010.rename(acs_2010_education, axis=1)\
                                   .drop(['state', 'county'], axis=1)\
                                   .astype(int)\
                                   .set_index('tract')\
                                   .apply(process_2010_education, axis=1)

    return pd.concat([race_2010, income_2010, education_2010], axis=1)

def load_tokens(tokens_file):
    '''
    Loads dictionary of API tokens.

    tokens_file (str): path to a JSON file containing API tokens
    '''
    with open(tokens_file,'r') as file:
        return json.load(file)

