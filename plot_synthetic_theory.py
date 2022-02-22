"""

Reference: A Shah, K Shanmugam, K Ahuja 
"Finding Valid Adjustments under Non-ignorability with Minimal DAG Knowledge,"
In International Conference on Artificial Intelligence and Statistics (AISTATS), 2022

Last updated: February 22, 2022
Code author: Abhin Shah

File name: plot_synthetic_theory.py

Description: Code to generate the plots to validate theoretical results on synthetic dataset

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
  number_dimensions_list = [2,7,12]
  number_repetitions = args.nr

  effect_true = np.array(pd.read_csv('synthetic_theory/effect_true.csv',header = None))
  effect_x1x2x3 = np.array(pd.read_csv('synthetic_theory/effect_x1x2x3.csv',header = None))
  effect_x1 = np.array(pd.read_csv('synthetic_theory/effect_x1.csv',header = None))
  effect_x2 =np.array(pd.read_csv('synthetic_theory/effect_x2.csv',header = None))
  effect_x3 = np.array(pd.read_csv('synthetic_theory/effect_x3.csv',header = None))
  effect_x1x2 = np.array(pd.read_csv('synthetic_theory/effect_x1x2.csv',header = None))
  effect_x2x3 = np.array(pd.read_csv('synthetic_theory/effect_x2x3.csv',header = None))
  effect_x1x3 =np.array(pd.read_csv('synthetic_theory/effect_x1x3.csv',header = None))

  pvalue_x1_Gout = np.array(pd.read_csv('synthetic_theory/pvalue_x1_Gout.csv',header = None))
  pvalue_x2_Gout = np.array(pd.read_csv('synthetic_theory/pvalue_x2_Gout.csv',header = None))
  pvalue_x3_Gout = np.array(pd.read_csv('synthetic_theory/pvalue_x3_Gout.csv',header = None))
  pvalue_x1x2_Gout = np.array(pd.read_csv('synthetic_theory/pvalue_x1x2_Gout.csv',header = None))
  pvalue_x2x3_Gout = np.array(pd.read_csv('synthetic_theory/pvalue_x2x3_Gout.csv',header = None))
  pvalue_x1x3_Gout = np.array(pd.read_csv('synthetic_theory/pvalue_x1x3_Gout.csv',header = None))
  pvalue_x1x2x3_Gout = np.array(pd.read_csv('synthetic_theory/pvalue_x1x2x3_Gout.csv',header = None))

  pvalue_x1 = np.array(pd.read_csv('synthetic_theory/pvalue_x1.csv',header = None))
  pvalue_x2 = np.array(pd.read_csv('synthetic_theory/pvalue_x2.csv',header = None))
  pvalue_x3 = np.array(pd.read_csv('synthetic_theory/pvalue_x3.csv',header = None))
  pvalue_x1x2 = np.array(pd.read_csv('synthetic_theory/pvalue_x1x2.csv',header = None))
  pvalue_x2x3 = np.array(pd.read_csv('synthetic_theory/pvalue_x2x3.csv',header = None))
  pvalue_x1x3 = np.array(pd.read_csv('synthetic_theory/pvalue_x1x3.csv',header = None))
  pvalue_x1x2x3 = np.array(pd.read_csv('synthetic_theory/pvalue_x1x2x3.csv',header = None))

  avg_ate_error_x1x2x3 = np.mean(np.abs(effect_x1x2x3-effect_true), axis=0).reshape(1, len(number_dimensions_list))
  avg_ate_error_x1 = np.mean(np.abs(effect_x1-effect_true), axis=0).reshape(1, len(number_dimensions_list))
  avg_ate_error_x2 = np.mean(np.abs(effect_x2-effect_true), axis=0).reshape(1, len(number_dimensions_list))
  avg_ate_error_x3 = np.mean(np.abs(effect_x3-effect_true), axis=0).reshape(1, len(number_dimensions_list))
  avg_ate_error_x1x2 = np.mean(np.abs(effect_x1x2-effect_true), axis=0).reshape(1, len(number_dimensions_list))
  avg_ate_error_x2x3 = np.mean(np.abs(effect_x2x3-effect_true), axis=0).reshape(1, len(number_dimensions_list))
  avg_ate_error_x1x3 = np.mean(np.abs(effect_x1x3-effect_true), axis=0).reshape(1, len(number_dimensions_list))

  std_ate_error_x1x2x3 = np.std(np.abs(effect_x1x2x3-effect_true), axis=0).reshape(1, len(number_dimensions_list))/np.sqrt(number_repetitions)
  std_ate_error_x1 = np.std(np.abs(effect_x1-effect_true), axis=0).reshape(1, len(number_dimensions_list))/np.sqrt(number_repetitions)
  std_ate_error_x2 = np.std(np.abs(effect_x2-effect_true), axis=0).reshape(1, len(number_dimensions_list))/np.sqrt(number_repetitions)
  std_ate_error_x3 = np.std(np.abs(effect_x3-effect_true), axis=0).reshape(1, len(number_dimensions_list))/np.sqrt(number_repetitions)
  std_ate_error_x1x2 = np.std(np.abs(effect_x1x2-effect_true), axis=0).reshape(1, len(number_dimensions_list))/np.sqrt(number_repetitions)
  std_ate_error_x2x3 = np.std(np.abs(effect_x2x3-effect_true), axis=0).reshape(1, len(number_dimensions_list))/np.sqrt(number_repetitions)
  std_ate_error_x1x3 = np.std(np.abs(effect_x1x3-effect_true), axis=0).reshape(1, len(number_dimensions_list))/np.sqrt(number_repetitions)

  # figure 1
  avg_ate_errors = pd.DataFrame(np.concatenate((avg_ate_error_x1x2x3,avg_ate_error_x1x2,avg_ate_error_x2x3,avg_ate_error_x1x3,avg_ate_error_x1,avg_ate_error_x2,avg_ate_error_x3), axis = 0),columns=number_dimensions_list)
  avg_ate_errors.index = ['$\{x_1,x_2,x_3\}$','$\{x_1,x_2\}$', '$\{x_2,x_3\}$', '$\{x_1,x_3\}$', '$\{x_1\}$','$\{x_2\}$','$\{x_3\}$']
  avg_ate_errors.columns = [2*dim+1 for dim in number_dimensions_list]

  std_ate_errors = pd.DataFrame(np.concatenate((std_ate_error_x1x2x3,std_ate_error_x1x2,std_ate_error_x2x3,std_ate_error_x1x3,std_ate_error_x1,std_ate_error_x2,std_ate_error_x3), axis = 0),columns=number_dimensions_list)
  std_ate_errors.index = ['$\{x_1,x_2,x_3\}$','$\{x_1,x_2\}$', '$\{x_2,x_3\}$', '$\{x_1,x_3\}$', '$\{x_1\}$','$\{x_2\}$','$\{x_3\}$']
  std_ate_errors.columns = [2*dim+1 for dim in number_dimensions_list]

  avg_ate_errors_dummy = avg_ate_errors
  avg_ate_errors_dummy = avg_ate_errors_dummy.drop(['$\{x_1,x_3\}$','$\{x_1\}$','$\{x_3\}$'])
  std_ate_errors_dummy = std_ate_errors
  std_ate_errors_dummy = std_ate_errors_dummy.drop(['$\{x_1,x_3\}$','$\{x_1\}$','$\{x_3\}$'])
  ax = avg_ate_errors_dummy.plot.bar(rot=0, yerr = std_ate_errors_dummy) 
  ax.set_xlabel("adjustment set",fontsize = 18)
  ax.set_ylabel("average ATE error",fontsize = 14)
  plt.xticks(fontsize=16)
  plt.yticks(fontsize=16)
  plt.tight_layout()
  ax.legend(["$d = 5$", "$d = 15$", "$d = 25$"],loc = 'upper right', fontsize = 16)
  ax.figure.savefig('synthetic_theory/synthetic_theory_avg_ate_errors_vs_adjustment_sets.pdf')

  avg_pvalue_x1x2x3 = np.mean(pvalue_x1x2x3, axis=0).reshape(1, len(number_dimensions_list))
  avg_pvalue_x1 = np.mean(pvalue_x1, axis=0).reshape(1, len(number_dimensions_list))
  avg_pvalue_x2 = np.mean(pvalue_x2, axis=0).reshape(1, len(number_dimensions_list))
  avg_pvalue_x3 = np.mean(pvalue_x3, axis=0).reshape(1, len(number_dimensions_list))
  avg_pvalue_x1x2 = np.mean(pvalue_x1x2, axis=0).reshape(1, len(number_dimensions_list))
  avg_pvalue_x2x3 = np.mean(pvalue_x2x3, axis=0).reshape(1, len(number_dimensions_list))
  avg_pvalue_x1x3 = np.mean(pvalue_x1x3, axis=0).reshape(1, len(number_dimensions_list))

  std_pvalue_x1x2x3 = np.std(pvalue_x1x2x3, axis=0).reshape(1, len(number_dimensions_list))/np.sqrt(number_repetitions)
  std_pvalue_x1 = np.std(pvalue_x1, axis=0).reshape(1, len(number_dimensions_list))/np.sqrt(number_repetitions)
  std_pvalue_x2 = np.std(pvalue_x2, axis=0).reshape(1, len(number_dimensions_list))/np.sqrt(number_repetitions)
  std_pvalue_x3 = np.std(pvalue_x3, axis=0).reshape(1, len(number_dimensions_list))/np.sqrt(number_repetitions)
  std_pvalue_x1x2 = np.std(pvalue_x1x2, axis=0).reshape(1, len(number_dimensions_list))/np.sqrt(number_repetitions)
  std_pvalue_x2x3 = np.std(pvalue_x2x3, axis=0).reshape(1, len(number_dimensions_list))/np.sqrt(number_repetitions)
  std_pvalue_x1x3 = np.std(pvalue_x1x3, axis=0).reshape(1, len(number_dimensions_list))/np.sqrt(number_repetitions)

  avg_pvalue_x1x2x3_Gout = np.mean(pvalue_x1x2x3_Gout, axis=0).reshape(1, len(number_dimensions_list))
  avg_pvalue_x1_Gout = np.mean(pvalue_x1_Gout, axis=0).reshape(1, len(number_dimensions_list))
  avg_pvalue_x2_Gout = np.mean(pvalue_x2_Gout, axis=0).reshape(1, len(number_dimensions_list))
  avg_pvalue_x3_Gout = np.mean(pvalue_x3_Gout, axis=0).reshape(1, len(number_dimensions_list))
  avg_pvalue_x1x2_Gout = np.mean(pvalue_x1x2_Gout, axis=0).reshape(1, len(number_dimensions_list))
  avg_pvalue_x2x3_Gout = np.mean(pvalue_x2x3_Gout, axis=0).reshape(1, len(number_dimensions_list))
  avg_pvalue_x1x3_Gout = np.mean(pvalue_x1x3_Gout, axis=0).reshape(1, len(number_dimensions_list))

  std_pvalue_x1x2x3_Gout = np.std(pvalue_x1x2x3_Gout, axis=0).reshape(1, len(number_dimensions_list))/np.sqrt(number_repetitions)
  std_pvalue_x1_Gout = np.std(pvalue_x1_Gout, axis=0).reshape(1, len(number_dimensions_list))/np.sqrt(number_repetitions)
  std_pvalue_x2_Gout = np.std(pvalue_x2_Gout, axis=0).reshape(1, len(number_dimensions_list))/np.sqrt(number_repetitions)
  std_pvalue_x3_Gout = np.std(pvalue_x3_Gout, axis=0).reshape(1, len(number_dimensions_list))/np.sqrt(number_repetitions)
  std_pvalue_x1x2_Gout = np.std(pvalue_x1x2_Gout, axis=0).reshape(1, len(number_dimensions_list))/np.sqrt(number_repetitions)
  std_pvalue_x2x3_Gout = np.std(pvalue_x2x3_Gout, axis=0).reshape(1, len(number_dimensions_list))/np.sqrt(number_repetitions)
  std_pvalue_x1x3_Gout = np.std(pvalue_x1x3_Gout, axis=0).reshape(1, len(number_dimensions_list))/np.sqrt(number_repetitions)

  avg_p_values_combined = pd.DataFrame(np.concatenate((avg_pvalue_x1x2x3_Gout,avg_pvalue_x1x2_Gout,avg_pvalue_x2x3_Gout,avg_pvalue_x2_Gout,avg_pvalue_x2,avg_pvalue_x2x3), axis = 0),columns=number_dimensions_list)
  avg_p_values_combined.index = ['$\{x_1,x_2,x_3\}$ in $\mathcal{G}_{-t}$','$\{x_1,x_2\}$ in $\mathcal{G}_{-t}$', '$\{x_2,x_3\}$ in $\mathcal{G}_{-t}$', '$\{x_2\}$ in $\mathcal{G}_{-t}$','$\{x_2\}$ in $\mathcal{G}$','$\{x_2,x_3\}$ in $\mathcal{G}$']
  avg_p_values_combined.columns = [2*dim+1 for dim in number_dimensions_list]

  std_p_values_combined = pd.DataFrame(np.concatenate((std_pvalue_x1x2x3_Gout,std_pvalue_x1x2_Gout,std_pvalue_x2x3_Gout,std_pvalue_x2_Gout,std_pvalue_x2,std_pvalue_x2x3), axis = 0),columns=number_dimensions_list)
  std_p_values_combined.index = ['$\{x_1,x_2,x_3\}$ in $\mathcal{G}_{-t}$','$\{x_1,x_2\}$ in $\mathcal{G}_{-t}$', '$\{x_2,x_3\}$ in $\mathcal{G}_{-t}$', '$\{x_2\}$ in $\mathcal{G}_{-t}$','$\{x_2\}$ in $\mathcal{G}$','$\{x_2,x_3\}$ in $\mathcal{G}$']
  std_p_values_combined.columns = [2*dim+1 for dim in number_dimensions_list]

  # figure 2
  ax = avg_p_values_combined.plot.bar(rot=0, yerr = std_p_values_combined)
  ax.set_xlabel("adjustment set",fontsize = 18)
  ax.set_ylabel("average $p_{value}$",fontsize = 14)
  plt.xticks(fontsize=12)
  plt.yticks(fontsize=14)
  plt.tight_layout()
  ax.legend(["$d = 5$", "$d = 15$", "$d = 25$"],loc = 'upper right', fontsize = 16)

  ax.figure.savefig('synthetic_theory/synthetic_theory_avg_p_values_vs_adjustment_sets.pdf')

  # figure 3
  succ_prob = pd.DataFrame(np.concatenate((np.sum(pvalue_x2 > 0.1 , axis = 0).reshape(1,-1)/number_repetitions,np.sum(pvalue_x2 > 0.2 , axis = 0).reshape(1,-1)/number_repetitions,np.sum(pvalue_x2 > 0.3 , axis = 0).reshape(1,-1)/number_repetitions,np.sum(pvalue_x2 > 0.4 , axis = 0).reshape(1,-1)/number_repetitions,np.sum(pvalue_x2 > 0.5 , axis = 0).reshape(1,-1)/number_repetitions), axis = 0),columns=number_dimensions_list)
  succ_prob.index = ['0.1','0.2', '0.3', '0.4', '0.5']
  succ_prob.columns = [2*dim+1 for dim in number_dimensions_list]
  ax = succ_prob.plot.bar(rot=0)
  ax.set_xlabel("$p_{value}$",fontsize = 18)
  ax.set_ylabel("average probability",fontsize = 14)
  ax.set_yticks([0.2,0.4,0.6,0.8,1.0], minor=False) 
  plt.xticks(fontsize=16)
  plt.yticks(fontsize=16)
  plt.tight_layout()
  ax.legend(["$d = 5$", "$d = 15$", "$d = 25$"],loc = 'upper right', fontsize = 16)
  ax.figure.savefig('synthetic_theory/synthetic_theory_success_prob_vs_p_values.pdf')


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--nr',
      help='number of repetitions',
      default=100,
      type=int)
  
  args = parser.parse_args()
  main(args)