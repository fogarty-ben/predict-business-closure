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
    ward (bool): if true, cta data is assumed to be ward level; if false, cta
        data is assumed to be zipcode level

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
