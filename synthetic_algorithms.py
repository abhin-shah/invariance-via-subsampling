"""

Reference: A Shah, K Shanmugam, K Ahuja 
"Finding Valid Adjustments under Non-ignorability with Minimal DAG Knowledge,"
In International Conference on Artificial Intelligence and Statistics (AISTATS), 2022

Last updated: February 22, 2022
Code author: Abhin Shah

File name: synthetic_algorithms.py

Description: Code to test Algorithms 1 and 2 on synthetic dataset

"""

import time
import pickle
import argparse
import multiprocessing
import numpy as np
import pandas as pd
from functools import partial
from sklearn.preprocessing import StandardScaler
from scipy.special import softmax, expit
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

@contextmanager
def poolcontext(*args, **kwargs):
  pool = multiprocessing.Pool(*args, **kwargs)
  yield pool
  pool.terminate()

def get_u(number_observations,number_dimensions):
  u1 = np.random.uniform(1.0,2.0,(number_observations,1))
  u2 = np.random.uniform(1.0,2.0,(number_observations,number_dimensions))
  u3 = np.random.uniform(1.0,2.0,(number_observations,number_dimensions))
  u4 = np.random.uniform(1.0,2.0,(number_observations,number_dimensions))
  return u1,u2,u3,u4

def get_theta(number_dimensions,use_t_in_e,number_environments):
  theta1 = np.random.uniform(1.0,2.0,(number_dimensions+1,1))
  theta2 = np.random.uniform(1.0,2.0,(number_dimensions*2+1,number_dimensions))
  theta3 = np.random.uniform(1.0,2.0,(number_dimensions*2,number_dimensions))
  theta4 = np.random.uniform(1.0,2.0,(number_dimensions*2+1,1))
  theta5 = np.random.uniform(1.0,2.0,(2,1))
  if use_t_in_e == 0:
    theta6 = np.concatenate((np.random.uniform(1.0,2.0,(1,1)), np.zeros((1,1)), np.random.uniform(-1.0,-2.0,(1,1))), axis = 1)
  else:
    theta6 = np.concatenate((np.random.uniform(1.0,2.0,(2,1)), np.zeros((2,1)), np.random.uniform(-1.0,-2.0,(2,1))), axis = 1)

  return theta1, theta2, theta3, theta4, theta5, theta6

def get_data(u1,u2,u3,u4,theta1,theta2,theta3,theta4,theta5,theta6,number_observations,number_dimensions,use_t_in_e,number_environments):
  x1 = np.dot(np.concatenate((u1,u2),axis=1), theta1) + 0.1* np.random.normal(0,1,(number_observations,1))
  x2 = np.dot(np.concatenate((x1,u2,u3),axis=1), theta2) + 0.1* np.random.normal(0,1,(number_observations,number_dimensions))
  x3 = np.dot(np.concatenate((u3,u4),axis=1), theta3) + 0.1 * np.random.normal(0,1,(number_observations,number_dimensions))
  t = np.random.binomial(1, expit(np.dot(np.concatenate((u1, x1), axis = 1), theta5)-np.mean(np.dot(np.concatenate((u1, x1), axis = 1), theta5))))
  if use_t_in_e == 0:
    probabilites = softmax(np.dot(x1, theta6)-np.mean(np.dot(x1, theta6), axis = 0), axis = 1)
  else:
    probabilites = softmax(np.dot(np.concatenate((x1, t), axis = 1), theta6)-np.mean(np.dot(np.concatenate((x1, t), axis = 1), theta6), axis = 0 ),  axis = 1)
  e = np.zeros((number_observations,1), dtype=int)
  for i,probability in enumerate(probabilites):
    e[i,:] = np.random.choice(np.arange(number_environments), p = probability)
  zeros =  np.zeros((number_observations,1))
  ones =  np.ones((number_observations,1))
  noise = 0.1 * np.random.normal(0,1,(number_observations,1))
  y = np.dot(np.concatenate((x2,u4,t),axis=1),theta4) + noise
  y_0 = np.dot(np.concatenate((x2,u4,zeros),axis=1),theta4) + noise
  y_1 = np.dot(np.concatenate((x2,u4,ones),axis=1),theta4) + noise

  return x1,x2,x3,t,e,y,y_0,y_1

def get_train_test_data(number_observations,number_dimensions,use_t_in_e,number_environments):
  u1,u2,u3,u4 = get_u(number_observations,number_dimensions)
  theta1, theta2, theta3, theta4, theta5, theta6 = get_theta(number_dimensions,use_t_in_e,number_environments)
  x1,x2,x3,t,e,y,y_0,y_1 = get_data(u1,u2,u3,u4,theta1,theta2,theta3,theta4,theta5,theta6,number_observations,number_dimensions,use_t_in_e,number_environments)
  true_ate = np.round(np.mean(y_1 - y_0),6)
  features_train, features_test, t_train, t_test, y_train, y_test, e_train, e_test, y_0_train, y_0_test, y_1_train, y_1_test = train_test_split(np.concatenate((x1,x2,x3),axis=1), t, y, e, y_0, y_1, test_size=0.2)
  x_train = pd.DataFrame(features_train)
  x_train.columns = list(map(str, range(2*number_dimensions+1)))
  x_train.columns = pd.MultiIndex.from_arrays([['x1' for i in range(1)] + ['x2' for i in range(number_dimensions)] + ['x3' for i in range(number_dimensions)], list(x_train.columns)]) 
  t_train = pd.Series(t_train[:,0])
  y_train = pd.Series(y_train[:,0])
  e_train = pd.Series(e_train[:,0])
  y_0_train = pd.Series(y_0_train[:,0])
  y_1_train = pd.Series(y_1_train[:,0])
  x_test = pd.DataFrame(features_test)
  x_test.columns = list(map(str, range(2*number_dimensions+1)))
  x_test.columns = pd.MultiIndex.from_arrays([['x1' for i in range(1)] + ['x2' for i in range(number_dimensions)] + ['x3' for i in range(number_dimensions)], list(x_test.columns)]) 
  t_test = pd.Series(t_test[:,0])
  y_test = pd.Series(y_test[:,0])
  e_test = pd.Series(e_test[:,0])
  y_0_test = pd.Series(y_0_test[:,0])
  y_1_test = pd.Series(y_1_test[:,0])
  return x_train, t_train, y_train, e_train, y_0_train, y_1_train, x_test, t_test, y_test, e_test, y_0_test, y_1_test, true_ate

def get_effect(x_train, t_train, y_train, x_test, t_test):
  std = Standardization(RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1]))
  std.fit(x_train, t_train, y_train)
  pop_outcome = std.estimate_population_outcome(x_test, t_test, agg_func="mean")
  return std.estimate_effect(pop_outcome[1], pop_outcome[0])

def powerset(s,number_dimensions):
  masks = [1 << i for i in range(2*number_dimensions)]
  for i in range(1,1 << 2*number_dimensions):
      yield [ss for mask, ss in zip(masks, s) if i & mask]

def get_all_subsets(number_dimensions):
  col = []
  for j in range(0,2*number_dimensions):
    col.append(str(j))
  return list(powerset(col, number_dimensions))

def get_correct_subset(i_subset):
  return [str(int(subset) + 1) for subset in i_subset]

def worker(subset,e,y_n,relevant_features,t):
  return RCoT(e,y_n,np.concatenate((relevant_features[subset].to_numpy(),t),axis=1))[0]

def main(args):

  starting_time = time.time()
  number_repetitions = args.nr
  number_observations = args.no
  use_t_in_e = args.use_t_in_e
  number_environments = args.ne
  number_dimensions_list = [1,2,3]
  p_thresholds = [0.1,0.2,0.3,0.4,0.5]

  exhaustive_ate_error = np.zeros((len(p_thresholds), number_repetitions,len(number_dimensions_list)))
  sparse_ate_error = np.zeros((len(p_thresholds), number_repetitions,len(number_dimensions_list)))
  baseline_ate_error = np.zeros((number_repetitions,len(number_dimensions_list)))
  irm_c_ate_error = np.zeros((number_repetitions,len(number_dimensions_list)))
  irm_t_ate_error = np.zeros((number_repetitions,len(number_dimensions_list)))
  oracle_ate_error = np.zeros((number_repetitions,len(number_dimensions_list)))

  set_start_method("spawn")
  for r_iter in range(number_repetitions):
    print(r_iter+1)
    for d_iter, number_dimensions in enumerate(number_dimensions_list):
      print(2*number_dimensions+1)
      x_train, t_train, y_train, e_train, y_0_train, y_1_train, x_test, t_test, y_test, e_test, y_0_test, y_1_test, true_ate = get_train_test_data(number_observations,number_dimensions,use_t_in_e,number_environments)
    
      baseline_ate_error[r_iter,d_iter] = np.abs(get_effect(x_train[['x1','x2','x3']], t_train, y_train, x_test[['x1','x2','x3']], t_test) - true_ate)
      oracle_ate_error[r_iter,d_iter] = np.abs(get_effect(x_train[['x2']], t_train, y_train, x_test[['x2']], t_test) - true_ate)
      
      x2_n = StandardScaler().fit_transform(np.array(x_train[['x2']]))
      x3_n = StandardScaler().fit_transform(np.array(x_train[['x3']]))
      relevant_features = pd.DataFrame(np.concatenate((x2_n,x3_n),axis=1))
      relevant_features.columns = list(map(str, range(2*number_dimensions)))
      y_n = StandardScaler().fit_transform(np.array(y_train).reshape(-1, 1))
      t = np.array(t_train).reshape(-1, 1)
      e = np.array(e_train).reshape(-1, 1)

      irm_features_control, _ = get_irm_features(x_train[['x2','x3']], t_train, y_train, e_train, number_environments, 0, x_train.columns.droplevel()[1:], args)
      irm_features_treatment, _ = get_irm_features(x_train[['x2','x3']], t_train, y_train, e_train, number_environments, 1, x_train.columns.droplevel()[1:], args)
      x_train.columns = x_train.columns.droplevel()
      x_test.columns = x_test.columns.droplevel()
      irm_c_ate_error[r_iter,d_iter] = np.abs(get_effect(x_train[irm_features_control], t_train, y_train, x_test[irm_features_control], t_test) - true_ate)
      irm_t_ate_error[r_iter,d_iter] = np.abs(get_effect(x_train[irm_features_treatment], t_train, y_train, x_test[irm_features_treatment], t_test) - true_ate)

      all_subsets = get_all_subsets(number_dimensions)
      all_p_values = []
      with get_context("spawn").Pool(processes = 40) as pool:
        all_p_values = pool.map(partial(worker, e=e, y_n=y_n, relevant_features=relevant_features, t=t), all_subsets)

      for p_iter, p_threshold in enumerate(p_thresholds):
        indices_above_threshold = [i for i in range(len(all_p_values)) if all_p_values[i] >= p_threshold]
        all_subset_above_threshold = list(map(all_subsets.__getitem__, indices_above_threshold))
        avg_ate_error = 0
        count = 0
        if len(all_subset_above_threshold):
          for relevant_subset in all_subset_above_threshold:
            correct_subset = get_correct_subset(relevant_subset)
            current_ate_error = np.abs(get_effect(x_train[correct_subset], t_train, y_train, x_test[correct_subset], t_test)-true_ate)
            exhaustive_ate_error[p_iter,r_iter,d_iter] += current_ate_error
            if len(correct_subset) == number_dimensions:
              count += 1
              sparse_ate_error[p_iter,r_iter,d_iter] += current_ate_error
          exhaustive_ate_error[p_iter,r_iter,d_iter] = exhaustive_ate_error[p_iter,r_iter,d_iter]/len(all_subset_above_threshold)
          if count > 0:
            sparse_ate_error[p_iter,r_iter,d_iter] = sparse_ate_error[p_iter,r_iter,d_iter] / count
    print(time.time() - starting_time)

  output = open('synthetic_algorithms/exhaustive_ate_error.pkl', 'wb')
  pickle.dump(exhaustive_ate_error, output)
  output.close()
  output = open('synthetic_algorithms/sparse_ate_error.pkl', 'wb')
  pickle.dump(sparse_ate_error, output)
  output.close()
  output = open('synthetic_algorithms/baseline_ate_error.pkl', 'wb')
  pickle.dump(baseline_ate_error, output)
  output.close()
  output = open('synthetic_algorithms/irm_c_ate_error.pkl', 'wb')
  pickle.dump(irm_c_ate_error, output)
  output.close()
  output = open('synthetic_algorithms/irm_t_ate_error.pkl', 'wb')
  pickle.dump(irm_t_ate_error, output)
  output.close()

  exhaustive_ate_error[exhaustive_ate_error == 0] = 'nan'

  avg_combined_ate_error = pd.DataFrame(np.concatenate(((np.mean(baseline_ate_error, axis=0)).reshape(1,-1),(np.mean(irm_c_ate_error, axis=0)).reshape(1,-1),(np.mean(irm_t_ate_error, axis=0)).reshape(1,-1),np.nanmean(exhaustive_ate_error, axis=1))),columns=number_dimensions_list)
  avg_combined_ate_error.index = ['$\{x_1,x_2,x_3\}$','IRM-c','IRM-t','0.1','0.2', '0.3', '0.4', '0.5']
  avg_combined_ate_error.columns = [2*dim+1 for dim in number_dimensions_list]
  avg_combined_ate_error = avg_combined_ate_error.replace(np.nan, 0)

  std_combined_ate_error = pd.DataFrame(np.concatenate((((np.std(baseline_ate_error, axis=0)).reshape(1,-1))/np.sqrt(number_repetitions),((np.std(irm_c_ate_error, axis=0)).reshape(1,-1))/np.sqrt(number_repetitions),((np.std(irm_t_ate_error, axis=0)).reshape(1,-1))/np.sqrt(number_repetitions),(np.nanstd(exhaustive_ate_error, axis=1))/np.sqrt(np.count_nonzero(~np.isnan(exhaustive_ate_error), axis=1)))),columns=number_dimensions_list)
  std_combined_ate_error.index = ['$\{x_1,x_2,x_3\}$','IRM-c','IRM-t','0.1','0.2', '0.3', '0.4', '0.5']
  std_combined_ate_error.columns = [2*dim+1 for dim in number_dimensions_list]
  std_combined_ate_error = std_combined_ate_error.replace(np.nan, 0)

  print(avg_combined_ate_error)
  print(std_combined_ate_error)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--nr',
      help='number of repetitions',
      default=100,
      type=int)
  parser.add_argument(
      '--no',
      help='number of observations',
      default=50000,
      type=int)
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
