"""

Reference: A Shah, K Shanmugam, K Ahuja 
"Finding Valid Adjustments under Non-ignorability with Minimal DAG Knowledge,"
In International Conference on Artificial Intelligence and Statistics (AISTATS), 2022

Last updated: February 22, 2022
Code author: Abhin Shah

File name: ihdp.py

Description: Code to test Algorithms 1 and 2 on IHDP dataset

"""

import time
import argparse
import multiprocessing
import pyreadr
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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, RidgeCV
from causallib.estimation import IPW, Standardization, StratifiedStandardization

from irm import get_irm_features
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
from rpy2.robjects.packages import importr
importr('RCIT')
RCoT = ro.r('RCoT')

def get_ihdp_compressed(args):
	data = pd.read_csv("https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/IHDP/csv/ihdp_npci_1.csv", header = None)
	col =  ["treatment", "y_factual", "y_cfactual", "mu0", "mu1" ,]
	for i in range(1,26):
		col.append(str(i))
	data.columns = col

	true_ate = np.mean(data['mu1']-data['mu0'])

	ihdp_data = pyreadr.read_r('data/ihdp.RData')['ihdp']
	ihdp_data = ihdp_data[(ihdp_data['treat'] != 1) | (ihdp_data['momwhite'] != 0)].reset_index(drop=True)
	ihdp_data = ihdp_data.drop(['momwhite', 'momblack', 'momhisp'], axis=1)

	ihdp_data_compressed = ihdp_data.copy()
	cols_to_norm = ['bw','b.head','preterm','birth.o','nnhealth','momage']
	ihdp_data_compressed[cols_to_norm] = ihdp_data_compressed[cols_to_norm].apply(lambda x: (x - x.mean()) / (x.std()))
	ihdp_data_compressed.columns = ['treatment','birth-weight','head-circumference','pre-term','birth-order','neonatal','age','sex','twin','married','edu-left-hs','edu-hs','edu-sc','cig','first-born','alcohol','drugs','working','prenatal','ark','ein','har','mia','pen','tex','was']
	ihdp_data_compressed['y_factual'] = data['y_factual']
	ihdp_data_compressed['y_cfactual'] = data['y_cfactual']
	ihdp_data_compressed['ite'] = (ihdp_data_compressed['y_factual'] - ihdp_data_compressed['y_cfactual'])*(2*ihdp_data_compressed['treatment']-1)
	cols_list = ihdp_data_compressed.columns.tolist()
	ihdp_data_compressed = ihdp_data_compressed[cols_list[1:26] + cols_list[0:1] + cols_list[26:]]

	variable_dict = {'birth-weight':"1",
					'head-circumference':"2",
					'pre-term': "3",
					'birth-order': "4",
					'neonatal':"5",
					'age':"6",
					'sex':"7",
					'twin':"8",
					'married':"9",
					'edu-left-hs':"10",
					'edu-hs':"10",
					'edu-sc':"10",
					'cig':"11",
					'first-born':"12",
					'alcohol':"13",
					'drugs':"14",
					'working':"15",
					'prenatal':"16",
					'ark':"17",
					'ein':"17",
					'har':"17",
					'mia':"17",
					'pen':"17",
					'tex':"17",
					'was':"17"}
	ihdp_data_compressed.columns = ["1","2","3","4","5","6","7","8","9","10","10","10","11","12","13","14","15","16","17","17","17","17","17","17","17","treatment", "y_factual", "y_cfactual","ite"] 

	return ihdp_data_compressed, variable_dict, true_ate

def get_x_t(ihdp_data_compressed, variable_dict, x_t_name):
	x_t = ihdp_data_compressed[variable_dict[x_t_name]].to_numpy().reshape(-1,1)
	return x_t

def get_effect(data_train, data_test, control_list):
	std = Standardization(RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1]))
	std.fit(data_train[control_list], data_train['treatment'], data_train['y_factual'])
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

def get_train_test_indices(ihdp_data_compressed):
	idx = np.random.permutation(ihdp_data_compressed.shape[0])
	train_idx = idx[:int(0.8 * ihdp_data_compressed.shape[0])]
	test_idx = idx[int(0.8 * ihdp_data_compressed.shape[0]):]

	return train_idx, test_idx

def get_all_subsets(number_dimensions, col):
	all_subsets = []
	for i in range(1,number_dimensions+1,1):
		for j in list(map(set, itertools.combinations(col, i))):
			all_subsets.append(sorted(list(j)))

	return all_subsets

def worker(subset,e,y_n,features,t):
	return RCoT(e,y_n,np.concatenate((features[subset].to_numpy(),t),axis=1))[0]

def main(args):
	starting_time = time.time()
	x_t_name_1 = args.xt_name_1
	number_repetitions = args.nr
	use_t_in_e = args.use_t_in_e
	number_environments = args.ne
	number_relevant_dimensions = args.nrd
	number_dimensions = 9
	p_thresholds = [0.1,0.2,0.3,0.4,0.5]

	success_prob = np.zeros((len(p_thresholds), number_repetitions))

	baseline_ate_error = np.zeros((1, number_repetitions))
	oracle_ate_error = np.zeros((1, number_repetitions))

	exhaustive_ate_error_1 = np.ones((len(p_thresholds), number_repetitions))
	sparse_ate_error_1 = np.ones((len(p_thresholds), number_repetitions))
	irm_c_ate_error_1 = np.zeros((1, number_repetitions))
	irm_t_ate_error_1 = np.zeros((1, number_repetitions))

	set_start_method("spawn")

	ihdp_data_compressed, variable_dict, true_ate = get_ihdp_compressed(args)
	x_t_1 = get_x_t(ihdp_data_compressed, variable_dict, x_t_name_1)
	col_to_keep_1 = ['3','2','4','5','6','7','14','16','17']
	all_subsets_1 = get_all_subsets(number_dimensions, col_to_keep_1)
	ihdp_data_compressed['e_1'] = get_environments(x_t_1,ihdp_data_compressed['treatment'].to_numpy().reshape(-1,1),use_t_in_e,number_environments)

	ihdp_data_irm = ihdp_data_compressed.copy()
	ihdp_data_irm.columns = pd.MultiIndex.from_arrays([ihdp_data_irm.columns.tolist(), list(np.arange(len(ihdp_data_irm.columns.tolist()[:-5]))) + ihdp_data_irm.columns.tolist()[-5:]]) 
	ihdp_data_irm.columns = ihdp_data_irm.columns.droplevel()
	cols_for_irm_1 = [1,2,3,4,5,6,15,17,18,19,20,21,22,23,24]

	for r_iter in range(number_repetitions):
		print(r_iter+1)
		train_idx, test_idx = get_train_test_indices(ihdp_data_compressed)	
		baseline_ate_error[0,r_iter] = np.abs(get_effect(ihdp_data_compressed.iloc[train_idx], ihdp_data_compressed.iloc[test_idx], [variable_dict[x_t_name_1]] + col_to_keep_1)-true_ate)
		oracle_ate_error[0,r_iter] = np.abs(get_effect(ihdp_data_compressed.iloc[train_idx], ihdp_data_compressed.iloc[test_idx], ['2', '4', '14', '16', '17'])-true_ate)
		features_1 = ihdp_data_compressed.iloc[train_idx][col_to_keep_1]

		irm_features_c_1, _ = get_irm_features(ihdp_data_irm.iloc[train_idx][cols_for_irm_1], ihdp_data_irm.iloc[train_idx]['treatment'], ihdp_data_irm.iloc[train_idx]['y_factual'], ihdp_data_irm.iloc[train_idx]['e_1'], number_environments, 0, pd.Series(cols_for_irm_1),args)
		irm_features_t_1, _ = get_irm_features(ihdp_data_irm.iloc[train_idx][cols_for_irm_1], ihdp_data_irm.iloc[train_idx]['treatment'], ihdp_data_irm.iloc[train_idx]['y_factual'], ihdp_data_irm.iloc[train_idx]['e_1'], number_environments, 1, pd.Series(cols_for_irm_1),args)
		irm_c_ate_error_1[0,r_iter] = np.abs(get_effect(ihdp_data_irm.iloc[train_idx], ihdp_data_irm.iloc[test_idx], irm_features_c_1.tolist())-true_ate)
		irm_t_ate_error_1[0,r_iter] = np.abs(get_effect(ihdp_data_irm.iloc[train_idx], ihdp_data_irm.iloc[test_idx], irm_features_t_1.tolist())-true_ate)

		y_n = StandardScaler().fit_transform(ihdp_data_compressed.iloc[train_idx]['y_factual'].to_numpy().reshape(-1,1))
		e_1 = ihdp_data_compressed.iloc[train_idx]['e_1'].to_numpy().reshape(-1,1)
		t = ihdp_data_compressed.iloc[train_idx]['treatment'].to_numpy().reshape(-1,1)

		all_p_values_1 = []
		# for subset in all_subsets:
		# 	all_p_values.append(RCoT(e,y_n,np.concatenate((features[subset].to_numpy(),t),axis=1))[0])
		with get_context("spawn").Pool(processes = 40) as pool:
			all_p_values_1 = pool.map(partial(worker, e=e_1, y_n=y_n, features=features_1, t=t), all_subsets_1)
		
		p_oracle = all_p_values_1[all_subsets_1.index(['14', '16','17', '2', '4'])]
		for p_iter, p_threshold in enumerate(p_thresholds):
			if p_oracle >= p_threshold:
				success_prob[p_iter,r_iter] += 1

			indices_above_threshold_1 = [i for i in range(len(all_p_values_1)) if all_p_values_1[i] >= p_threshold]
			all_subset_above_threshold_1 = list(map(all_subsets_1.__getitem__, indices_above_threshold_1))
			count_1 = 0
			if len(all_subset_above_threshold_1):
				for relevant_subset_1 in all_subset_above_threshold_1:
					current_ate_error_1 = np.abs(get_effect(ihdp_data_compressed.iloc[train_idx], ihdp_data_compressed.iloc[test_idx], relevant_subset_1)-true_ate)
					exhaustive_ate_error_1[p_iter,r_iter] += current_ate_error_1
					if len(relevant_subset_1) <= number_relevant_dimensions:
						count_1 += 1
						sparse_ate_error_1[p_iter,r_iter] += current_ate_error_1

				exhaustive_ate_error_1[p_iter,r_iter] = exhaustive_ate_error_1[p_iter,r_iter]/len(all_subset_above_threshold_1)
				if count_1 > 0:
					sparse_ate_error_1[p_iter,r_iter] = sparse_ate_error_1[p_iter,r_iter] / count_1

		print(time.time() - starting_time)

	np.savetxt('ihdp/exhaustive_ate_error_1.csv', exhaustive_ate_error_1, delimiter=",")
	np.savetxt('ihdp/sparse_ate_error_1.csv', sparse_ate_error_1, delimiter=",")
	np.savetxt('ihdp/irm_c_ate_error_1.csv', irm_c_ate_error_1, delimiter=",")
	np.savetxt('ihdp/irm_t_ate_error_1.csv', irm_t_ate_error_1, delimiter=",")
	np.savetxt('ihdp/baseline_ate_error.csv', baseline_ate_error, delimiter=",")
	np.savetxt('ihdp/oracle_ate_error.csv', oracle_ate_error, delimiter=",")
	np.savetxt('ihdp/success_prob.csv', success_prob, delimiter=",")

	exhaustive_ate_error_1[exhaustive_ate_error_1 == 0] = 'nan'
	avg_exhaustive_ate_error_1  = np.nanmean(exhaustive_ate_error_1, axis = 1)
	std_exhaustive_ate_error_1  = np.nanstd(exhaustive_ate_error_1, axis = 1)/np.sqrt(np.count_nonzero(~np.isnan(exhaustive_ate_error_1), axis=1))

	avg_baseline_ate_error = np.mean(baseline_ate_error, axis = 1)
	std_baseline_ate_error = np.std(baseline_ate_error, axis = 1)/np.sqrt(number_repetitions)
	
	avg_oracle_ate_error = np.mean(oracle_ate_error, axis = 1)
	std_oracle_ate_error = np.std(oracle_ate_error, axis = 1)/np.sqrt(number_repetitions)
	
	avg_irm_c_ate_error_1 = np.mean(irm_c_ate_error_1, axis = 1)
	std_irm_c_ate_error_1 = np.std(irm_c_ate_error_1, axis = 1)/np.sqrt(number_repetitions)

	avg_irm_t_ate_error_1 = np.mean(irm_t_ate_error_1, axis = 1)
	std_irm_t_ate_error_1 = np.std(irm_t_ate_error_1, axis = 1)/np.sqrt(number_repetitions)

	avg_ate_errors = pd.DataFrame(np.concatenate((avg_baseline_ate_error,avg_oracle_ate_error,avg_irm_c_ate_error_1,avg_irm_t_ate_error_1), axis = 0))
	avg_ate_errors.index = ['baseline','oracle', 'IRM-c-1', 'IRM-t-1']

	std_ate_errors = pd.DataFrame(np.concatenate((std_baseline_ate_error,std_oracle_ate_error,std_irm_c_ate_error_1,std_irm_t_ate_error_1), axis = 0))
	std_ate_errors.index = ['baseline','oracle', 'IRM-c-1', 'IRM-t-1']

	print(avg_ate_errors)
	print(avg_exhaustive_ate_error_1)
	print(std_ate_errors)
	print(std_exhaustive_ate_error_1)

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
		'--xt_name_1',
		help='x_t variable 1',
		default='birth-weight',
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