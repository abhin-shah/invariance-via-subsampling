"""

Reference: A Shah, K Shanmugam, K Ahuja 
"Finding Valid Adjustments under Non-ignorability with Minimal DAG Knowledge,"
In International Conference on Artificial Intelligence and Statistics (AISTATS), 2022

Last updated: February 22, 2022
Code author: Abhin Shah

File name: plot_synthetic_high_dimension.py

Description: Code to generate the plots to test Algorithm 2 on synthetic dataset in high dimension

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
  number_dimensions_list = [12,22,32]
  number_repetitions = args.nr

  effect_true = np.array(pd.read_csv('synthetic_high_dimension/effect_true.csv',header = None))
  effect_baseline = np.array(pd.read_csv('synthetic_high_dimension/effect_baseline.csv',header = None))
  effect_irm_control = np.array(pd.read_csv('synthetic_high_dimension/effect_irm_control.csv',header = None))
  effect_irm_treatment =np.array(pd.read_csv('synthetic_high_dimension/effect_irm_treatment.csv',header = None))

  avg_ate_error_baseline = np.mean(np.abs(effect_baseline-effect_true), axis=0).reshape(1, len(number_dimensions_list))
  avg_ate_error_irm_c = np.mean(np.abs(effect_irm_control-effect_true), axis=0).reshape(1, len(number_dimensions_list))
  avg_ate_error_irm_t = np.mean(np.abs(effect_irm_treatment-effect_true), axis=0).reshape(1, len(number_dimensions_list))

  std_ate_error_baseline = np.std(np.abs(effect_baseline-effect_true), axis=0).reshape(1, len(number_dimensions_list))/np.sqrt(number_repetitions)
  std_ate_error_irm_c = np.std(np.abs(effect_irm_control-effect_true), axis=0).reshape(1, len(number_dimensions_list))/np.sqrt(number_repetitions)
  std_ate_error_irm_t = np.std(np.abs(effect_irm_treatment-effect_true), axis=0).reshape(1, len(number_dimensions_list))/np.sqrt(number_repetitions)

  avg_ate_errors = pd.DataFrame(np.concatenate((avg_ate_error_baseline,avg_ate_error_irm_c,avg_ate_error_irm_t), axis = 0),columns=number_dimensions_list)
  avg_ate_errors.index = ['baseline','IRM-c', 'IRM-t']
  avg_ate_errors.columns = [2*dim+1 for dim in number_dimensions_list]

  std_ate_errors = pd.DataFrame(np.concatenate((std_ate_error_baseline,std_ate_error_irm_c,std_ate_error_irm_t), axis = 0),columns=number_dimensions_list)
  std_ate_errors.index = ['baseline','IRM-c', 'IRM-t']
  std_ate_errors.columns = [2*dim+1 for dim in number_dimensions_list]

  ax = avg_ate_errors.plot.bar(rot=0, yerr = std_ate_errors) 
  ax.set_xlabel("methods",fontsize=24)
  ax.set_ylabel("average ATE error",fontsize=24)
  plt.xticks(fontsize=24)
  plt.yticks(fontsize=24)
  plt.tight_layout()
  lgd = ax.legend(["$d = 25$", "$d = 45$", "$d = 65$"], loc = 'upper right', fontsize = 18)
  ax.figure.savefig('synthetic_high_dimension/synthetic_high_dimension_avg_ate_errors.pdf')


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--nr',
      help='number of repetitions',
      default=100,
      type=int)
  
  args = parser.parse_args()
  main(args)
