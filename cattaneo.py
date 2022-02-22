"""

Reference: A Shah, K Shanmugam, K Ahuja 
"Finding Valid Adjustments under Non-ignorability with Minimal DAG Knowledge,"
In International Conference on Artificial Intelligence and Statistics (AISTATS), 2022

Last updated: February 22, 2022
Code author: Abhin Shah

File name: cattaneo.py

Description: Code to test Algorithms 1 and 2 on Cattaneo dataset

"""

import time
import argparse
import multiprocessing
import random
import itertools
import numpy as np
import pandas as pd
from functools import partial
from sklearn.preprocessing import StandardScaler
from scipy.special import expit, softmax
from contextlib import contextmanager
from multiprocessing import get_context
from multiprocessing import set_start_method
from sklearn.linear_model import LogisticRegression, RidgeCV
from causallib.estimation import IPW, Standardization, StratifiedStandardization

from irm import get_irm_features

import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
from rpy2.robjects.packages import importr
importr('RCIT')
RCoT = ro.r('RCoT')

def get_cattaneo_compressed():
	cattaneo_ori = pd.read_stata('data/cattaneo2.dta')
	cattaneo_ori['mmarried'] = cattaneo_ori['mmarried'].replace('married', 1)
	cattaneo_ori['mmarried'] = cattaneo_ori['mmarried'].replace('notmarried', 0)
	cattaneo_ori['mbsmoke'] = cattaneo_ori['mbsmoke'].replace('smoker', 1)
	cattaneo_ori['mbsmoke'] = cattaneo_ori['mbsmoke'].replace('nonsmoker', 0)
	cattaneo_ori['fbaby'] = cattaneo_ori['fbaby'].replace('Yes', 1)
	cattaneo_ori['fbaby'] = cattaneo_ori['fbaby'].replace('No', 0)
	cattaneo_ori['prenatal1'] = cattaneo_ori['prenatal1'].replace('Yes', 1)
	cattaneo_ori['prenatal1'] = cattaneo_ori['prenatal1'].replace('No', 0)
	cattaneo_ori = cattaneo_ori.drop(['msmoke', 'lbweight'], axis=1)
	cols_to_norm = ['mage','medu','fage','fedu','nprenatal','monthslb','order','prenatal','birthmonth']
	cattaneo_norm = cattaneo_ori.copy()
	cattaneo_norm[cols_to_norm] = cattaneo_norm[cols_to_norm].apply(lambda x: (x - x.mean()) / (x.std()))
	cols_list = cattaneo_norm.columns.to_list()
	cattaneo_norm = cattaneo_norm[cols_list[7:14] + cols_list[17:19] + cols_list[1:7] + cols_list[15:17] + cols_list[19:] + cols_list[14:15]+ cols_list[0:1]]
	cattaneo_norm.columns = cattaneo_norm.columns.to_list()[:-2] + ['treatment','y']
	variable_dict = {'mage':"1",
					'medu':"2",
					'fage': "3",
					'fedu': "4",
					'nprenatal':"5",
					'prenatal1':"6",
					'prenatal':"7",
					'monthslb':"8",
					'order':"9",
					'birthmonth':"10",
					'mmarried':"11",
					'mhisp':"12",
					'mrace':"13",
					'frace':"14",
					'fhisp':"15",
					'foreign':"16",
					'alcohol':"17",
					'deadkids':"18",
					'fbaby':"19"}
	cattaneo_compressed = cattaneo_norm.copy()
	cattaneo_compressed.columns = list(variable_dict.values()) + ['treatment','y']
	cat_columns = cattaneo_compressed.select_dtypes(['category']).columns
	cattaneo_compressed[cat_columns] = cattaneo_compressed[cat_columns].apply(lambda x: x.cat.codes)
	return cattaneo_compressed, variable_dict

def get_x_t(cattaneo_compressed, variable_dict, x_t_name):
	x_t = cattaneo_compressed[variable_dict[x_t_name]].to_numpy().reshape(-1,1)
	return x_t

def get_effect(data_train, data_test, control_list):
	std = Standardization(RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1]))
	std.fit(data_train[control_list], data_train['treatment'], data_train['y'])
	pop_outcome = std.estimate_population_outcome(data_test[control_list], data_test['treatment'], agg_func="mean")
	return std.estimate_effect(pop_outcome[1], pop_outcome[0])

def get_environments(x_t,t,use_t_in_e,number_environments):
	if use_t_in_e == 0:
		theta = np.concatenate((np.random.uniform(1.0,2.0,(1,1)), np.zeros((1,1)), np.random.uniform(-1.0,-2.0,(1,1))), axis = 1)
		probabilites = softmax(np.dot(x_t, theta)-np.mean(np.dot(x_t, theta), axis = 0), axis = 1)
	else:
		theta = np.concatenate((np.random.uniform(1.0,2.0,(2,1)), np.zeros((2,1)), np.random.uniform(-1.0,-2.0,(2,1))), axis = 1)
		probabilites = softmax(np.dot(np.concatenate((x_t, t), axis = 1), theta)-np.mean(np.dot(np.concatenate((x_t, t), axis = 1), theta),  axis = 0 ),  axis = 1)
	e = np.zeros(x_t.shape, dtype=int)
	for i,probability in enumerate(probabilites):
		e[i,:] = np.random.choice(np.arange(number_environments), p = probability)
	return e

def get_train_test_indices(cattaneo_compressed):
	idx = np.random.permutation(cattaneo_compressed.shape[0])
	train_idx = idx[:int(0.8 * cattaneo_compressed.shape[0])]
	test_idx = idx[int(0.8 * cattaneo_compressed.shape[0]):]

	return train_idx, test_idx

def get_all_subsets(number_relevant_dimensions, number_dimensions):
	col = []
	for i in range(2,number_dimensions+1,1):
		col.append(str(i))
	all_subsets = []
	for i in range(1,number_relevant_dimensions+1,1):
		for j in list(map(set, itertools.combinations(col, i))):
			all_subsets.append(list(j))

	return col, all_subsets 

def worker(subset,e,y_n,features,t):
	return RCoT(e,y_n,np.concatenate((features[subset].to_numpy(),t),axis=1))[0]

def main(args):
	starting_time = time.time()
	x_t_name = args.xt_name
	number_repetitions = args.nr
	number_environments = args.ne
	use_t_in_e = args.use_t_in_e
	number_relevant_dimensions = args.nrd
	p_thresholds = [0.1,0.2,0.3,0.4,0.5]

	sparse_ate_error = np.ones((len(p_thresholds), number_repetitions))
	baseline_ate_error = np.zeros((1, number_repetitions))
	irm_c_ate_error = np.zeros((1, number_repetitions))
	irm_t_ate_error = np.zeros((1, number_repetitions))

	cattaneo_compressed, variable_dict = get_cattaneo_compressed()
	number_dimensions = len(variable_dict)
	x_t = get_x_t(cattaneo_compressed, variable_dict, x_t_name)
	cattaneo_compressed['e'] = get_environments(x_t,cattaneo_compressed['treatment'].to_numpy().reshape(-1,1),use_t_in_e,number_environments)

	col, all_subsets = get_all_subsets(number_relevant_dimensions, number_dimensions)

	set_start_method("spawn")

	cols_for_irm = ['2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19']

	cattaneo_compressed[['y_norm']] = cattaneo_compressed[['y']].apply(lambda x: (x - x.mean()) / (x.std()))

	for r_iter in range(number_repetitions):
		print(r_iter+1)
		train_idx, test_idx = get_train_test_indices(cattaneo_compressed)
		baseline_ate_error[0,r_iter] = get_effect(cattaneo_compressed.iloc[train_idx], cattaneo_compressed.iloc[test_idx], np.unique(cattaneo_compressed.columns.tolist()[:-4]))[0]
		features = cattaneo_compressed.iloc[train_idx][col]

		irm_features_c, _ = get_irm_features(cattaneo_compressed.iloc[train_idx][cols_for_irm], cattaneo_compressed.iloc[train_idx]['treatment'], cattaneo_compressed.iloc[train_idx]['y_norm'], cattaneo_compressed.iloc[train_idx]['e'], number_environments, 0, pd.Series(cols_for_irm), args)
		irm_features_t, _ = get_irm_features(cattaneo_compressed.iloc[train_idx][cols_for_irm], cattaneo_compressed.iloc[train_idx]['treatment'], cattaneo_compressed.iloc[train_idx]['y_norm'], cattaneo_compressed.iloc[train_idx]['e'], number_environments, 1, pd.Series(cols_for_irm), args)
		irm_c_ate_error[0,r_iter] = get_effect(cattaneo_compressed.iloc[train_idx], cattaneo_compressed.iloc[test_idx], irm_features_c.tolist())
		irm_t_ate_error[0,r_iter] = get_effect(cattaneo_compressed.iloc[train_idx], cattaneo_compressed.iloc[test_idx], irm_features_t.tolist())

		y_n = StandardScaler().fit_transform(cattaneo_compressed.iloc[train_idx]['y'].to_numpy().reshape(-1,1))
		e = cattaneo_compressed.iloc[train_idx]['e'].to_numpy().reshape(-1,1)
		t = cattaneo_compressed.iloc[train_idx]['treatment'].to_numpy().reshape(-1,1)

		all_p_values = []
		# for subset in all_subsets:
		# 	all_p_values.append(RCoT(e,y_n,np.concatenate((features[subset].to_numpy(),t),axis=1))[0])
		with get_context("spawn").Pool(processes = 40) as pool:
			all_p_values = pool.map(partial(worker, e=e, y_n=y_n, features=features, t=t), all_subsets)

		for p_iter, p_threshold in enumerate(p_thresholds):
			indices_above_threshold = [i for i in range(len(all_p_values)) if all_p_values[i] >= p_threshold]
			all_subset_above_threshold = list(map(all_subsets.__getitem__, indices_above_threshold))
			if len(all_subset_above_threshold):
				for relevant_subset in all_subset_above_threshold:
					sparse_ate_error[p_iter,r_iter] += get_effect(cattaneo_compressed.iloc[train_idx], cattaneo_compressed.iloc[test_idx], relevant_subset)

			sparse_ate_error[p_iter,r_iter] = sparse_ate_error[p_iter,r_iter]/len(all_subset_above_threshold)
		print(time.time() - starting_time)
	
	np.savetxt('cattaneo/sparse_ate_error.csv', sparse_ate_error, delimiter=",")
	np.savetxt('cattaneo/baseline_ate_error.csv', baseline_ate_error, delimiter=",")
	np.savetxt('cattaneo/irm_c_ate_error.csv', irm_c_ate_error, delimiter=",")
	np.savetxt('cattaneo/irm_t_ate_error.csv', irm_t_ate_error, delimiter=",")

	sparse_ate_error[sparse_ate_error == 0] = 'nan'

	avg_baseline_ate_error = np.mean(baseline_ate_error, axis = 1)
	std_baseline_ate_error = np.std(baseline_ate_error, axis = 1)/np.sqrt(number_repetitions)
	avg_sparse_ate_error = np.nanmean(sparse_ate_error, axis = 1)
	std_sparse_ate_error = np.nanstd(sparse_ate_error, axis = 1)/np.sqrt(np.count_nonzero(~np.isnan(sparse_ate_error), axis=1))
	avg_irm_c_ate_error = np.mean(irm_c_ate_error, axis = 1)
	std_irm_c_ate_error = np.std(irm_c_ate_error, axis = 1)/np.sqrt(number_repetitions)
	avg_irm_t_ate_error = np.mean(irm_t_ate_error, axis = 1)
	std_irm_t_ate_error = np.std(irm_t_ate_error, axis = 1)/np.sqrt(number_repetitions)

	avg_ate_errors = pd.DataFrame(np.concatenate((avg_baseline_ate_error,avg_irm_c_ate_error,avg_irm_t_ate_error), axis = 0))
	avg_ate_errors.index = ['baseline','IRM-c', 'IRM-t']

	std_ate_errors = pd.DataFrame(np.concatenate((std_baseline_ate_error,std_irm_c_ate_error,std_irm_t_ate_error), axis = 0))
	std_ate_errors.index = ['baseline','IRM-c', 'IRM-t']

	print(avg_ate_errors)
	print(avg_sparse_ate_error)
	print(std_ate_errors)
	print(std_sparse_ate_error)

	
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--nr',
		help='number of repetitions',
		default=100,
		type=int)
	parser.add_argument(
		'--nrd',
		help='number of relevant dimensions',
		default=5,
		type=int)
	parser.add_argument(
		'--xt_name',
		help='x_t variable',
		default='mage',
		type=str)
	parser.add_argument(
		'--use_t_in_e',
		help='indicator for whether t should be used to generate e',
		default=1,
		type=int)
	parser.add_argument(
		'--ne',
		help='number of environments',
		default=3,
		type=int)
	parser.add_argument(
		'--number_IRM_iterations',
		help='number of IRM iterations',
		default=15000,
		type=int)

	args = parser.parse_args()
	main(args)