"""

Reference: A Shah, K Shanmugam, K Ahuja 
"Finding Valid Adjustments under Non-ignorability with Minimal DAG Knowledge,"
In International Conference on Artificial Intelligence and Statistics (AISTATS), 2022

Last updated: February 22, 2022
Code author: Abhin Shah

File name: plot_cattaneo.py

Description: Code to generate the plots for Cattaneo dataset

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

	baseline_ate_error = np.array(pd.read_csv('cattaneo/baseline_ate_error.csv',header = None))
	sparse_ate_error = np.array(pd.read_csv('cattaneo/sparse_ate_error.csv',header = None))
	irm_c_ate_error = np.array(pd.read_csv('cattaneo/irm_c_ate_error.csv',header = None))
	irm_t_ate_error = np.array(pd.read_csv('cattaneo/irm_t_ate_error.csv',header = None))

	avg_baseline_ate_error = np.mean(baseline_ate_error, axis = 1)
	std_baseline_ate_error = np.std(baseline_ate_error, axis = 1)/np.sqrt(number_repetitions)

	avg_irm_c_ate_error = np.mean(irm_c_ate_error, axis = 1)
	std_irm_c_ate_error = np.std(irm_c_ate_error, axis = 1)/np.sqrt(number_repetitions)
	avg_irm_t_ate_error = np.mean(irm_t_ate_error, axis = 1)
	std_irm_t_ate_error = np.std(irm_t_ate_error, axis = 1)/np.sqrt(number_repetitions)

	sparse_ate_error[sparse_ate_error == 0] = 'nan'

	avg_sparse_ate_error  = np.nanmean(sparse_ate_error, axis = 1)
	avg_sparse_ate_error = pd.DataFrame(avg_sparse_ate_error)
	avg_sparse_ate_error.index = ['0.1','0.2', '0.3', '0.4', '0.5']

	std_sparse_ate_error  = np.nanstd(sparse_ate_error, axis = 1)/np.sqrt(np.count_nonzero(~np.isnan(sparse_ate_error), axis=1))
	std_sparse_ate_error = pd.DataFrame(std_sparse_ate_error)
	std_sparse_ate_error.index = ['0.1','0.2', '0.3', '0.4', '0.5']

	line1 = plt.errorbar(avg_sparse_ate_error.index.to_list(), [avg_baseline_ate_error[0]]*len(avg_sparse_ate_error), yerr = [std_baseline_ate_error[0]]*len(avg_sparse_ate_error), label = 'baseline')
	line2 = plt.errorbar(avg_sparse_ate_error.index.to_list(), list(avg_sparse_ate_error[0]), yerr = list(std_sparse_ate_error[0]), label = 'sparse')
	line3 = plt.errorbar(avg_sparse_ate_error.index.to_list(), [avg_irm_c_ate_error[0]]*len(avg_sparse_ate_error), yerr = [std_irm_c_ate_error[0]]*len(avg_sparse_ate_error), label = 'IRM-c')
	line4 = plt.errorbar(avg_sparse_ate_error.index.to_list(), [avg_irm_t_ate_error[0]]*len(avg_sparse_ate_error), yerr = [std_irm_t_ate_error[0]]*len(avg_sparse_ate_error), label = 'IRM-t')

	ax = plt.gca()
	handles, labels = ax.get_legend_handles_labels()
	handles = [h[0] for h in handles]
	ax.legend(handles,labels, loc = 'upper right', fontsize = 16)
	ax.set_xlabel("$p_{value}$",fontsize = 24)
	ax.set_ylabel("average ATE",fontsize = 24)
	plt.xticks(fontsize=24)
	ax.set_yticks([-250,-240,-230,-220,-210,-200], minor=False) 
	plt.yticks(fontsize=24)
	plt.tight_layout()
	ax.figure.savefig('cattaneo/cattaneo_ate_errors_vs_p_values.pdf')
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