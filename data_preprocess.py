'''
pre-processing functions

(dataset-specific)
'''
import numpy as np
import pandas as pd

def pre_pipeline_clean(df):

	# Data preparation
	## Generate outcome variable
	df['time_till_funded'] = (df.datefullyfunded - df.date_posted).apply(lambda x: x.days)
	df['not_funded_wi_60d'] = np.where(df.time_till_funded > 60, 1, 0)

	# keep only top values of selected columns
	top_school_city = ['Los Angeles', 'Chicago', 'Houston', 'Brooklyn', 'Bronx']
	top_school_state = ['CA', 'NY', 'TX', 'FL', 'IL']
	top_school_dist = ['Los Angeles Unif Sch Dist', 'New York City Dept Of Ed',
	                   'Philadelphia City School Dist', 'Miami-dade Co Public Sch Dist', 
	                   'Clark Co School District', 'Charlotte-mecklenburg Sch Dist',
	                   'San Francisco Unified Sch Dist']
	top_school_county = ['Los Angeles', 'Orange', 'Cook', 'Harris', 'Kings (Brooklyn)', 'Alameda']

	keep_top_values(df, 'school_city', top_school_city)
	keep_top_values(df, 'school_state', top_school_state)
	keep_top_values(df, 'school_district', top_school_dist)
	keep_top_values(df, 'school_county', top_school_county)

	return df

def preprocess(X, y):
	'''
	preprocess data
	'''
	# drop rows with NA values for certain columns
	df = pd.concat([X, y], axis=1)
	cols_to_drop_na_rows = ['school_district', 
	                'primary_focus_subject', 
	                'primary_focus_area', 
	                'resource_type', 
	                'grade_level']
	df.dropna(axis=0, how='any', subset=cols_to_drop_na_rows, inplace=True)
	y = df.iloc[:,-1]
	X = df.iloc[:,0:-1]

	# fill specific columns with median
	X['students_reached'].fillna(value=X['students_reached'].median())

	# encode

	# scale

def predictors_to_discretize():
	rv = {'students_reached': (
								[0, 20, 30, 90, float('inf')],
								['<20', '20-30', '30-90', '>90']),
		   'total_price_including_optional_support':(
								[0, 300, 600, 900, 1200, float('inf')],
								['<300', '300-600', '600-900', '900-1200', '>1200'])}
	return rv

def keep_top_values(df, col_name, top_values):
    '''
    col_name: column name
    top_values: list of top values of the column to keep
    '''
    df[col_name].where(df[col_name].isin(top_values), other="other", inplace=True)
    