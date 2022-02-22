"""

Reference: A Shah, K Shanmugam, K Ahuja 
"Finding Valid Adjustments under Non-ignorability with Minimal DAG Knowledge,"
In International Conference on Artificial Intelligence and Statistics (AISTATS), 2022

Last updated: February 22, 2022
Code author: Abhin Shah

File name: plot_ihdp.py

Description: Code to generate the plots for IHDP dataset

"""

import time
import argparse
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

def main(args):
	number_repetitions = args.nr
	
	exhaustive_ate_error_1 = np.array(pd.read_csv('ihdp/exhaustive_ate_error_1.csv',header = None))
	sparse_ate_error_1 = np.array(pd.read_csv('ihdp/sparse_ate_error_1.csv',header = None))
	
	irm_c_ate_error_1 = np.array(pd.read_csv('ihdp/irm_c_ate_error_1.csv',header = None))
	irm_t_ate_error_1 = np.array(pd.read_csv('ihdp/irm_t_ate_error_1.csv',header = None))
	
	baseline_ate_error = np.array(pd.read_csv('ihdp/baseline_ate_error.csv',header = None))
	oracle_ate_error = np.array(pd.read_csv('ihdp/oracle_ate_error.csv',header = None))
	success_prob = np.array(pd.read_csv('ihdp/success_prob.csv',header = None))

	# figure 1
	avg_success_prob = pd.DataFrame(np.sum(success_prob, axis = 1)/number_repetitions)
	avg_success_prob.index = ['0.1','0.2', '0.3', '0.4', '0.5']
	ax = avg_success_prob.plot.bar(rot=0, legend = False) 
	ax.set_xlabel("$p_{value}$",fontsize = 24)
	ax.set_ylabel("average probability",fontsize = 24)
	ax.set_yticks([0.2,0.4,0.6,0.8,1.0], minor=False) 
	plt.xticks(fontsize = 24)
	plt.yticks(fontsize = 24)
	plt.tight_layout()
	ax.figure.savefig('ihdp/ihdp_success_prob_vs_p_values.pdf')
	plt.show()

	# figure 2
	avg_baseline_ate_error = np.mean(baseline_ate_error, axis = 1)
	std_baseline_ate_error = np.std(baseline_ate_error, axis = 1)/np.sqrt(number_repetitions)

	avg_irm_c_ate_error_1 = np.mean(irm_c_ate_error_1, axis = 1)
	std_irm_c_ate_error_1 = np.std(irm_c_ate_error_1, axis = 1)/np.sqrt(number_repetitions)

	avg_irm_t_ate_error_1 = np.mean(irm_t_ate_error_1, axis = 1)
	std_irm_t_ate_error_1 = np.std(irm_t_ate_error_1, axis = 1)/np.sqrt(number_repetitions)

	avg_oracle_ate_error = np.mean(oracle_ate_error, axis = 1)
	std_oracle_ate_error = np.std(oracle_ate_error, axis = 1)/np.sqrt(number_repetitions)

	exhaustive_ate_error_1[exhaustive_ate_error_1 == 0] = 'nan'
	sparse_ate_error_1[sparse_ate_error_1 == 0] = 'nan'

	avg_exhaustive_ate_error_1  = np.nanmean(exhaustive_ate_error_1, axis = 1)
	avg_exhaustive_ate_error_1 = pd.DataFrame(avg_exhaustive_ate_error_1)
	avg_exhaustive_ate_error_1.index = ['0.1','0.2', '0.3', '0.4', '0.5']

	std_exhaustive_ate_error_1  = np.nanstd(exhaustive_ate_error_1, axis = 1)/np.sqrt(np.count_nonzero(~np.isnan(exhaustive_ate_error_1), axis=1))
	std_exhaustive_ate_error_1 = pd.DataFrame(std_exhaustive_ate_error_1)
	std_exhaustive_ate_error_1.index = ['0.1','0.2', '0.3', '0.4', '0.5']

	avg_sparse_ate_error_1  = np.nanmean(sparse_ate_error_1, axis = 1)
	avg_sparse_ate_error_1 = pd.DataFrame(avg_sparse_ate_error_1)
	avg_sparse_ate_error_1.index = ['0.1','0.2', '0.3', '0.4', '0.5']

	std_sparse_ate_error_1  = np.nanstd(sparse_ate_error_1, axis = 1)/np.sqrt(np.count_nonzero(~np.isnan(sparse_ate_error_1), axis=1))
	std_sparse_ate_error_1 = pd.DataFrame(std_sparse_ate_error_1)
	std_sparse_ate_error_1.index = ['0.1','0.2', '0.3', '0.4', '0.5']

	params = {'mathtext.default': 'regular' }          
	plt.rcParams.update(params)	
	line1 = plt.errorbar(avg_sparse_ate_error_1.index.to_list(), [avg_irm_c_ate_error_1[0]]*len(avg_sparse_ate_error_1), yerr = [std_irm_c_ate_error_1[0]]*len(avg_sparse_ate_error_1), label = 'IRM-c')
	line2 = plt.errorbar(avg_sparse_ate_error_1.index.to_list(), [avg_baseline_ate_error[0]]*len(avg_sparse_ate_error_1), yerr = [std_baseline_ate_error[0]]*len(avg_sparse_ate_error_1), label = 'baseline')
	line3 = plt.errorbar(avg_sparse_ate_error_1.index.to_list(), list(avg_exhaustive_ate_error_1[0]), yerr = list(std_exhaustive_ate_error_1[0]), label = 'exhaustive')
	line4 = plt.errorbar(avg_sparse_ate_error_1.index.to_list(), list(avg_sparse_ate_error_1[0]), yerr = list(std_sparse_ate_error_1[0]), label = 'sparse')
	line5 = plt.errorbar(avg_sparse_ate_error_1.index.to_list(), [avg_irm_t_ate_error_1[0]]*len(avg_sparse_ate_error_1), yerr = [std_irm_t_ate_error_1[0]]*len(avg_sparse_ate_error_1), label = 'IRM-t')
	
	ax = plt.gca()
	handles, labels = ax.get_legend_handles_labels()
	handles = [h[0] for h in handles]
	lgd = ax.legend(handles,labels, loc = 'upper left', fontsize = 16)
	ax.set_xlabel("$p_{value}$",fontsize = 24)
	ax.set_ylabel("average ATE error",fontsize = 24)
	plt.xticks(fontsize = 24)
	ax.set_yticks([0.05,0.10,0.15,0.20,0.25,0.30], minor=False) 
	plt.yticks(fontsize = 24)
	plt.tight_layout()
	ax.figure.savefig('ihdp/birth_weight_ihdp_avg_ate_errors_vs_p_values.pdf')
	plt.show()

	# figure 3
	line1 = plt.errorbar(avg_sparse_ate_error_1.index.to_list(), [avg_baseline_ate_error[0]]*len(avg_sparse_ate_error_1), yerr = [std_baseline_ate_error[0]]*len(avg_sparse_ate_error_1), label = 'baseline')
	line2 = plt.errorbar(avg_sparse_ate_error_1.index.to_list(), [avg_oracle_ate_error[0]]*len(avg_sparse_ate_error_1), yerr = [std_oracle_ate_error[0]]*len(avg_sparse_ate_error_1), label = 'oracle')

	ax = plt.gca()
	handles, labels = ax.get_legend_handles_labels()
	handles = [h[0] for h in handles]
	lgd = ax.legend(handles,labels, loc = 'upper left', fontsize = 16)
	ax.set_xlabel("$p_{value}$",fontsize = 24)
	ax.set_ylabel("average ATE error",fontsize = 24)
	plt.xticks(fontsize = 24)
	ax.set_yticks([0.05,0.10,0.15,0.20,0.25,0.30], minor=False) 
	plt.yticks(fontsize = 24)
	plt.tight_layout()
	ax.figure.savefig('ihdp/ihdp_avg_ate_errors_with_oracle_vs_p_values.pdf')
	plt.show()

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--nr',
		help='number of repetitions',
		default=100,
		type=int)

	args = parser.parse_args()
	main(args)