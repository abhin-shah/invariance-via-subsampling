"""

Reference: A Shah, K Shanmugam, K Ahuja 
"Finding Valid Adjustments under Non-ignorability with Minimal DAG Knowledge,"
In International Conference on Artificial Intelligence and Statistics (AISTATS), 2022

Last updated: February 22, 2022
Code author: Abhin Shah

File name: plot_synthetic_algorithms.py

Description: Code to generate the plots to test Algorithms 1 and 2 on synthetic dataset

"""

import time
import argparse
import random
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

def main(args):
  number_dimensions_list = [1,2,3]
  number_repetitions = args.nr
  pkl_file = open('synthetic_algorithms/exhaustive_ate_error.pkl', 'rb')
  exhaustive_ate_error = pickle.load(pkl_file)
  pkl_file.close()
  pkl_file = open('synthetic_algorithms/sparse_ate_error.pkl', 'rb')
  sparse_ate_error = pickle.load(pkl_file)
  pkl_file.close()
  pkl_file = open('synthetic_algorithms/baseline_ate_error.pkl', 'rb')
  baseline_ate_error = pickle.load(pkl_file)
  pkl_file.close()
  pkl_file = open('synthetic_algorithms/irm_c_ate_error.pkl', 'rb')
  irm_c_ate_error = pickle.load(pkl_file)
  pkl_file.close()
  pkl_file = open('synthetic_algorithms/irm_t_ate_error.pkl', 'rb')
  irm_t_ate_error = pickle.load(pkl_file)
  pkl_file.close()

  exhaustive_ate_error[exhaustive_ate_error == 0] = 'nan'

  avg_combined_ate_error = pd.DataFrame(np.concatenate(((np.mean(baseline_ate_error, axis=0)).reshape(1,-1),np.nanmean(exhaustive_ate_error, axis=1),(np.mean(irm_c_ate_error, axis=0)).reshape(1,-1),(np.mean(irm_t_ate_error, axis=0)).reshape(1,-1))),columns=number_dimensions_list)
  avg_combined_ate_error.index = ['baseline','0.1','0.2', '0.3', '0.4', '0.5','IRM-c','IRM-t']
  avg_combined_ate_error.columns = [2*dim+1 for dim in number_dimensions_list]
  avg_combined_ate_error = avg_combined_ate_error.replace(np.nan, 0)

  std_combined_ate_error = pd.DataFrame(np.concatenate((((np.std(baseline_ate_error, axis=0)).reshape(1,-1))/np.sqrt(number_repetitions),(np.nanstd(exhaustive_ate_error, axis=1))/np.sqrt(np.count_nonzero(~np.isnan(exhaustive_ate_error), axis=1)),((np.std(irm_c_ate_error, axis=0)).reshape(1,-1))/np.sqrt(number_repetitions),((np.std(irm_t_ate_error, axis=0)).reshape(1,-1))/np.sqrt(number_repetitions))),columns=number_dimensions_list)
  std_combined_ate_error.index = ['baseline','0.1','0.2', '0.3', '0.4', '0.5','IRM-c','IRM-t']
  std_combined_ate_error.columns = [2*dim+1 for dim in number_dimensions_list]
  std_combined_ate_error = std_combined_ate_error.replace(np.nan, 0)

  ax = avg_combined_ate_error.plot.bar(rot=0, yerr = std_combined_ate_error) 
  ax.set_xlabel("$p_{value}$",fontsize=24)
  ax.set_ylabel("average ATE error",fontsize=24)
  plt.xticks(fontsize=18)
  plt.yticks(fontsize=24)
  plt.tight_layout()
  lgd = ax.legend(["$d = 3$", "$d = 5$", "$d = 7$"], loc = 'upper right', fontsize = 18)
  ax.figure.savefig('synthetic_algorithms/synthetic_algorithms_avg_exhaustive_ate_errors_vs_p_values.pdf')

  sparse_ate_error[sparse_ate_error == 0] = 'nan'

  avg_combined_exhaustive_ate_error = pd.DataFrame(np.concatenate(((np.mean(baseline_ate_error, axis=0)).reshape(1,-1),np.nanmean(sparse_ate_error, axis=1),(np.mean(irm_c_ate_error, axis=0)).reshape(1,-1),(np.mean(irm_t_ate_error, axis=0)).reshape(1,-1))),columns=number_dimensions_list)
  avg_combined_exhaustive_ate_error.index = ['baseline','0.1','0.2', '0.3', '0.4', '0.5','IRM-c','IRM-t']
  avg_combined_exhaustive_ate_error.columns = [2*dim+1 for dim in number_dimensions_list]
  avg_combined_exhaustive_ate_error = avg_combined_exhaustive_ate_error.replace(np.nan, 0)

  std_combined_exhaustive_ate_error = pd.DataFrame(np.concatenate((((np.std(baseline_ate_error, axis=0)).reshape(1,-1))/np.sqrt(number_repetitions),(np.nanstd(sparse_ate_error, axis=1))/np.sqrt(np.count_nonzero(~np.isnan(sparse_ate_error), axis=1)),((np.std(irm_c_ate_error, axis=0)).reshape(1,-1))/np.sqrt(number_repetitions),((np.std(irm_t_ate_error, axis=0)).reshape(1,-1))/np.sqrt(number_repetitions))),columns=number_dimensions_list)
  std_combined_exhaustive_ate_error.index = ['baseline','0.1','0.2', '0.3', '0.4', '0.5','IRM-c','IRM-t']
  std_combined_exhaustive_ate_error.columns = [2*dim+1 for dim in number_dimensions_list]
  std_combined_exhaustive_ate_error = std_combined_exhaustive_ate_error.replace(np.nan, 0)

  ax = avg_combined_exhaustive_ate_error.plot.bar(rot=0, yerr = std_combined_exhaustive_ate_error)
  ax.set_xlabel("$p_{value}$",fontsize=24)
  ax.set_ylabel("average ATE error",fontsize=24)
  plt.xticks(fontsize=18)
  plt.yticks(fontsize=24)
  plt.tight_layout()
  lgd = ax.legend(["$d = 3$", "$d = 5$", "$d = 7$"],loc = 'upper right', fontsize = 18)
  ax.figure.savefig('synthetic_algorithms/synthetic_algorithms_avg_sparse_ate_errors_vs_p_values.pdf')

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--nr',
      help='number of repetitions',
      default=100,
      type=int)
  
  args = parser.parse_args()
  main(args)
