import pandas as pd
import license_clean

def link_zbp_licenses(zbp, licenses):
    '''
    Links Census data on Zip Code Business Patterns with business licenses based
    on year and zipcodes.

    Inputs:
    zpb (pandas dataframe): a set of zip code bsuiness patterns data
    licenses (pandas dataframe): a set of business licenses data

    Returns: pandas dataframe
    '''
    #may need to subtract one year based on when data is from
    zbp['year_as_date'] = pd.to_datetime(zbp.year.astype(str), format='%Y')
    zbp, buckets = license_clean.create_time_buckets(zbp, {'years': 2}, 'year_as_date',
                                            '2002-01-01')
    zbp = zbp.drop(['year_as_date', 'year'], axis=1)
    zbp = zbp.groupby(['zipcode', 'time_period'])\
             .agg('mean')\
             .reset_index()
    
    zbp['merge_col'] = zbp.time_period.astype(str) + '-' + zbp.zipcode.astype(str)
    zbp = zbp.drop('time_period', axis=1)
    licenses['merge_col'] = licenses.time_period.astype(str) + '-' + licenses.zip_code.astype(str)

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
    period_startyr_map = licenses.groupby('time_period')\
                                 .agg({'min_start_date': 'min'})\
                                 ['min_start_date']\
                                 .dt.year
    licenses['period_startyr'] = licenses.time_period\
                                         .map(period_startyr_map)

    pre_2010_lcs_mask = licenses.period_startyr < 2010
    pre_2010 = pd.merge(licenses[pre_2010_lcs_mask], census_2000, how='left',
                        left_on='census_tract', right_on='tract')

    post_2010_lcs_mask = licenses.period_startyr >= 2010
    post_2010 = pd.merge(licenses[post_2010_lcs_mask], census_2010, how='left',
                         left_on='census_tract', right_on='tract')

    return pre_2010.append(post_2010, ignore_index=True)\
                   .drop('period_startyr', axis=1)

def link_cta_licenses(cta, licenses, months):
    '''
    Links 'El' station ridership with business license data based on a given
    time length.

    Inputs:
    cta (pandas dataframe): a set of cta ridership data
    licenses (pandas dataframe): a set of business licenses data
    months (int): number of months to aggregate cta ridership data over
    ward (bool): if true, cta data is assumed to be ward level; if false, cta
        data is assumed to be zipcode level

    Returns: pandas dataframe
    '''
    licenses['exp_month_year'] = licenses['max_expiration_date'].dt.to_period('M')

    cta = cta.groupby('Ward')\
             [['month_year', 'monthtotal', 'avg_weekday_rides']]\
             .rolling(window=months, on='month_year')\
             .mean()\
             .droplevel(1)\
             .reset_index()\
             .rename({'monthtotal': 'monthavg_last{}'.format(months),
                      'avg_weekday_rides': 'avg_weekday_rides_last{}'.format(months)},
                     axis=1)
    cta['merge_col'] = cta['Ward'].astype(str) + '_' + cta.month_year.astype(str)

    licenses['merge_col'] = (licenses['ward'].astype(str) + '_' +
                             licenses.exp_month_year.astype(str))

    return pd.merge(licenses, cta, how='left', on='merge_col')\
             .drop(['exp_month_year', 'merge_col', 'Ward'], axis=1)

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
    licenses['exp_month_year'] = licenses['max_expiration_date'].dt.to_period('M')

    real_estate = real_estate.groupby('RegionName')\
                             [['Month', 'MedianValuePerSqfeet_Zip']]\
                             .rolling(window=months, on='Month')\
                             .mean()\
                             .droplevel(1)\
                             .reset_index()

    real_estate['merge_col'] = real_estate['RegionName'].astype(str) + '_' + real_estate.Month.astype(str)

    licenses['merge_col'] = (licenses['zip_code'].astype(str) + '_' +
                             licenses.exp_month_year.astype(str))

    return pd.merge(licenses, real_estate, how='left', on='merge_col')\
             .drop(['exp_month_year', 'merge_col', 'RegionName', 'Month'], axis=1)

def link_gdp_licenses(gdp, licenses):
    '''
    Links Chicago GDP data with business license data.
    
    Inputs:
    gdp (pandas dataframe): a set of gdp data
    licenses (pandas dataframe): a set of business licenses data
    '''
    #GDP DATA PROBABLY NEEDS TO BE GDP GROWTH DATA
    gdp['year_as_date'] = pd.to_datetime(gdp.Year.astype(str), format='%Y')
    ump, buckets = license_clean.create_time_buckets(gdp, {'years': 2}, 'year_as_date',
                                            '2002-01-01')
    gdp = gdp.drop(['year_as_date', 'Year', 'GDP_billion_dollars'], axis=1)
    gdp = gdp.groupby(['time_period'])\
             .agg('mean')


    return pd.merge(licenses, gdp, how='left', on='time_period')

def link_ump_licenses(ump, licenses):
    '''
    Links Chicago unemployment rate data with business license data.
    
    Inputs:
    ump (pandas dataframe): a set of gdp data
    licenses (pandas dataframe): a set of business licenses data
    '''
    #GDP DATA PROBABLY NEEDS TO BE GDP GROWTH DATA
    ump['year_as_date'] = pd.to_datetime(ump.Year.astype(str), format='%Y')
    ump, buckets = license_clean.create_time_buckets(ump, {'years': 2}, 'year_as_date',
                                            '2002-01-01')
    ump = ump.drop(['year_as_date', 'Year'], axis=1)
    ump = ump.groupby(['time_period'])\
             .agg('mean')\
             .rename({'Annual': 'umpavg_timepd'}, axis=1)

    return pd.merge(licenses, ump, how='left', on='time_period')