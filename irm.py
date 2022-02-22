"""

Reference: A Shah, K Shanmugam, K Ahuja 
"Finding Valid Adjustments under Non-ignorability with Minimal DAG Knowledge,"
In International Conference on Artificial Intelligence and Statistics (AISTATS), 2022

Last updated: February 22, 2022
Code author: Abhin Shah

File name: irm.py

Description: Code to obtain the features selected by IRM in Algorithm 2

"""

import numpy as np
import torch
from torch.autograd import grad
from sklearn.cluster import KMeans

class InvariantRiskMinimization(object):
    def __init__(self, environments, args):
        best_reg = 0
        best_err = 1e100
        self.phi = 0
        self.dimension_x = 0
        self.flag = False

        x_val = environments[-1][0]
        y_val = environments[-1][1]

        for lr in [1e-2, 1e-3]:
          for reg in [1e-3, 1e-1]:
            self.train(environments[:-1], args, reg=reg, lr=lr)
            err = (x_val @ self.solution() - y_val).pow(2).mean().item()

            if err < best_err:
                best_err = err
                best_reg = reg
                best_lr = lr
                best_phi = self.phi.clone()
                self.flag = True

        if self.flag == False:
          print("Choosing some default parameters")
          best_err = err
          best_reg = reg
          best_lr = lr
          best_phi = self.phi.clone()
          self.flag = True

        self.phi = best_phi
        print("Best reg = "+str(best_reg))
        print("Best lr = "+str(best_lr))

    def train(self, environments, args, reg, lr):
        dim_x = environments[0][0].size(1)
        self.dimension_x = dim_x

        self.phi = torch.nn.Parameter(torch.eye(dim_x, dim_x))
        self.w = torch.ones(dim_x, 1)
        self.w.requires_grad = True

        opt = torch.optim.Adam([self.phi], lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=5000, gamma=0.5)

        loss = torch.nn.MSELoss()
        for iteration in range(args.number_IRM_iterations):
            penalty = 0
            error = 0
            for x_e, y_e in environments:
                error_e = loss(x_e @ self.phi @ self.w, y_e)
                gradient = grad(error_e, self.w, create_graph=True)
                penalty += gradient[0].pow(2).mean()
                error += error_e

            opt.zero_grad()
            (reg * error + (1 - reg) * penalty).backward()
            opt.step()
            scheduler.step()

    def solution(self):
        return self.phi @ self.w

def envs_irm_T(X_train, y_train, E, number_environments):
    """Compute the environments variable required by the InvariantRiskMinimization class for IRM_T / IRM_2
    Args:
    - X_train: training features
    - y_train: training observed outcome
    - E: training environments
    - number_environments: numbe of environments
    Returns:
    - environments: the environments variable required by the InvariantRiskMinimization class for IRM_T / IRM_2
    """
    X_train = X_train.astype(np.float32)
    y_train = y_train.astype(np.float32)

    environments = []

    for i in range(number_environments):
        E_index = np.where(E == i)[0]
        X = X_train[E_index,:]
        y = y_train[E_index,:]
        X_ = torch.from_numpy(X)
        y_ = torch.from_numpy(y)

        ones_training = torch.ones(X.shape[0], 1)
        features = torch.cat((ones_training, X_), 1)
        
        environments.append((features,y_))
    return environments

def IRM_T_block(environments, args):
    irm = InvariantRiskMinimization(environments, args)  
    irm_coeff = irm.solution()

    return irm_coeff

def get_irm_features(x_train, t_train, y_train, e_train, number_environments, group_indicator, feature_indices, args):
    T_group = np.where(t_train.to_numpy().reshape(-1, 1) == group_indicator)[0]
    environments_group = envs_irm_T(x_train.to_numpy()[T_group,:], y_train.to_numpy().reshape(-1, 1)[T_group,:], e_train.to_numpy().reshape(-1, 1)[T_group,:], number_environments)
    irm_group_coeff  = IRM_T_block(environments_group, args)
    irm_group_coeff = irm_group_coeff.detach().numpy()[1:]
    kmeans = KMeans(n_clusters=2, random_state=0).fit(np.absolute(irm_group_coeff))
    which_cluster = np.argmax(kmeans.cluster_centers_)
    
    return feature_indices[kmeans.labels_ == which_cluster], irm_group_coeff
