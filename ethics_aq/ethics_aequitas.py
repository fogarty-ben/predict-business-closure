import pandas as pd
import seaborn as sns
import numpy as np
from aequitas.group import Group
from aequitas.bias import Bias
from aequitas.fairness import Fairness
from aequitas.plotting import Plot
import sys
import warnings; warnings.simplefilter('ignore')
import matplotlib.pyplot as plt
# %matplotlib inline



def run_aequitas(predictions_data_path):
	'''
	Check for False negative rate, chances of certain group missing out on assistance using aequitas toolkit
	The functions transform the data to make it aequitas complaint and checks for series of bias and fairness metrics
	Input: model prediction path for the selected model (unzip the selected file to run)
	Output: plots saved in charts folder
	'''

	best_model_pred = pd.read_csv(predictions_data_path) 

	# Transform data for aquetias module compliance 
	aqc = ['Other','White','African American', 'Asian','Hispanic','American Indian']
	aqcol = ['White alone_scale', 'Black/AfAmer alone_scale','AmInd/Alaskn alone_scale','Asian alone_scale','HI alone_scale','Some other race alone_scale','Hispanic or Latino_scale']
	display(aqcol)
	aqcol_label = ['no_renew_nextpd','pred_class_10%','Median household income (1999 dollars)_scale'] + aqcol
	aqus = best_model_pred[aqcol_label]
	print('Creating classes for racial and income distribution','\n')

	# Convert to binary
	bin_var = ['no_renew_nextpd','pred_class_10%',]
	for var in bin_var:
	    aqus[var] = np.where(aqus[var] == True, 1, 0)
	# Rename
	aqus.rename(columns={'no_renew_nextpd': 'label_value', 'pred_class_10%': 'score'}, inplace=True)

	print('Define majority rule defined on relative proportion of the class','\n')
	aqus['race'] = aqus[aqcol].idxmax(axis=1)
	# Use quantile income distribution
	aqus['income'] = pd.qcut(aqus['Median household income (1999 dollars)_scale'], 3, labels=["rich", "median", "poor"])

	# Final form
	aqus.drop(aqcol, axis=1, inplace=True)
	aqus.drop(['Median household income (1999 dollars)_scale'], axis=1, inplace=True)
	aq = aqus.reset_index()
	aq.rename(columns={'index': 'entity_id'}, inplace=True)
	aq['race']=aq['race'].replace({'Some other race alone_scale': 'Other', 'White alone_scale': 'White', 
	                    'Black/AfAmer alone_scale': 'African American', 'Asian alone_scale': 'Asian',
	                   'HI alone_scale': 'Hispanic', 'AmInd/Alaskn alone_scale': 'American Indian'})

	# Consolidate types
	aq['income'] = aq['income'].astype(object)
	aq['entity_id'] = aq['entity_id'].astype(object)
	aq['score'] = aq['score'].astype(object)
	aq['label_value'] = aq['label_value'].astype(object)

	# Distribuion of categories
	aq_palette = sns.diverging_palette(225, 35, n=2)
	by_race = sns.countplot(x="race", data=aq[aq.race.isin(aqc)])
	by_race.set_xticklabels(by_race.get_xticklabels(), rotation=40, ha="right")
	plt.savefig('charts/Racial distribution in data.png')


	# Primary distribuion against score
	aq_palette = sns.diverging_palette(225, 35, n=2)
	by_race = sns.countplot(x="race", hue="score", 
	          data=aq[aq.race.isin(aqc)], palette=aq_palette)
	by_race.set_xticklabels(by_race.get_xticklabels(), rotation=40, ha="right")
	# Race
	plt.savefig('charts/race_score.png')
	# Income
	by_inc = sns.countplot(x="income", hue="score", data=aq, palette=aq_palette)
	plt.savefig('charts/income_score.png')

	# Set Group 
	g = Group()
	xtab, _ = g.get_crosstabs(aq)

	# False Negative Rates
	aqp = Plot()
	fnr = aqp.plot_group_metric(xtab, 'fnr', min_group_size=0.05)
	p = aqp.plot_group_metric_all(xtab, metrics=['ppr','pprev','fnr','fpr'], ncols=4)
	p.savefig('charts/eth_metrics.png')

	# Bias with respect to white rich category
	b = Bias()
	bdf = b.get_disparity_predefined_groups(xtab, original_df=aq, ref_groups_dict={'race':'White', 'income':'rich'},
	                                        alpha=0.05, mask_significance=True)
	bdf.style
	calculated_disparities = b.list_disparities(bdf)
	disparity_significance = b.list_significance(bdf)
	aqp.plot_disparity(bdf, group_metric='fpr_disparity', attribute_name='race', significance_alpha=0.05)
	plt.savefig('charts/disparity.png')

	# Fairness 
	hbdf = b.get_disparity_predefined_groups(xtab, original_df=aq, 
                                         ref_groups_dict={'race':'African American', 'income':'poor'},
                                         alpha=0.05,mask_significance=False)
	majority_bdf = b.get_disparity_major_group(xtab, original_df=aq, mask_significance=True)
	min_metric_bdf = b.get_disparity_min_metric(df=xtab, original_df=aq)
	f = Fairness()
	fdf = f.get_group_value_fairness(bdf)
	parity_detrminations = f.list_parities(fdf)
	gaf = f.get_group_attribute_fairness(fdf)
	gof = f.get_overall_fairness(fdf)
	z = aqp.plot_fairness_group(fdf, group_metric='ppr')
	plt.savefig('charts/fairness_overall.png')
	fg = aqp.plot_fairness_group_all(fdf, ncols=5, metrics = "all")
	plt.savefig('charts/fairness_metrics.png')

	return None

