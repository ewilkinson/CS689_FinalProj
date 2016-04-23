import scipy.optimize as opt
import matplotlib.pyplot as plt

from curvature import CurvatureExp
from vehicle_model.vehicle_model import VehicleModel

from utility.convertors import *
from opt_funcs.penalty_funcs import *
from opt_funcs.simple_grad_descent import simple_grad
from opt_funcs.adagrad import adagrad
from opt_funcs.adadelta import adadelta

import copy
import time


##########################################
####       COMPARISON PARAMETERS      ####
##########################################
F_TOL = 1e-4

# algorith: scipy options
algorithms = {'L-BFGS-B': {'disp': False},
              simple_grad: {'disp': True, 'stepsize': 0.001},
              adagrad: {'maxiter': 10000, 'eta': 0.005},
              adadelta: {'maxiter': 10000, 'rho_dx': 0.8, 'rho_g': 0.8}}

##########################################
####     END COMPARISON PARAMETERS    ####
##########################################

# Create a vehicle at the origin 0,0,0
veh_model = VehicleModel(np.asarray([0, 0, 0]), 1.0, VehicleModel.EXACT)


# Create a psi function. This integrates over length L, contains N gaussian kernels
# and has an initial sigma value of whatever is specified in sigint
L = 50
desired_spacing = 0.25
N = 25
sigint = 3.0
initial_weights_zero = True
psi = CurvatureExp(N=N,
                   L=L,
                   sigint=sigint,
                   use_cache=True,
                   zero_weights=initial_weights_zero)


# Parameters related to optimization
t_points = np.asarray([[10, 5],
                       [15, 5],
                       [25, -10]], dtype=np.float64)
penalty_args = {'desired_spacing': desired_spacing,
                'global_desired_theta': np.pi / 8,
                'last_desired_theta': 0.0,
                'should_opt_sigma': False,
                'target_points': t_points}

# 200 is a good value for l2 weights
penalty_weights = {GLOBAL_DIR_PENALTY: 0.0,
                   LAST_DIR_PENALTY: 0.0,
                   L1_WEIGHTS: 1.0,
                   L2_WEIGHTS: 0.0,
                   L1_CURVATURE: 0.0,
                   L2_CURVATURE: 0.0,
                   L2_DK: 0.0,
                   L1_DK: 0.0,
                   TARGET_POINTS_PENALTY: 1.0}
penalty_funcs = {GLOBAL_DIR_PENALTY: global_dir_penalty_func,
                 LAST_DIR_PENALTY: last_dir_penalty_func,
                 L1_WEIGHTS: l1_penalty_weights,
                 L2_WEIGHTS: l2_penalty_weights,
                 L1_CURVATURE: l1_penalty_curvature,
                 L2_CURVATURE: l2_penalty_curvature,
                 L2_DK: l2_penalty_dk,
                 L1_DK: l1_penalty_dk,
                 TARGET_POINTS_PENALTY: target_points_penalty}

# check that every penalty weight has an associated penalty func
weight_set = set(penalty_weights.keys())
func_set = set(penalty_funcs.keys())
assert len(weight_set.intersection(func_set)) == len(weight_set)


def min_func(x, psi, penalty_weights, penalty_args):
    map_x_to_psi(psi, x, penalty_args['should_opt_sigma'])

    total_cost = 0

    # for each weight that is non zero evaluate the penalty and add to total cost
    for key in penalty_weights.keys():
        if penalty_weights[key] != 0:
            penalty_func = penalty_funcs[key]
            penalty_weight = penalty_weights[key]
            p_cost = penalty_weight * penalty_func(psi, veh_model, penalty_args)

            total_cost += p_cost

    return total_cost


for algo, options in algorithms.iteritems():
    t0 = time.clock()
    x0 = create_x_from_psi(psi, penalty_args['should_opt_sigma'])
    res = opt.minimize(min_func, x0,
                       args=(copy.deepcopy(psi), penalty_weights, penalty_args),
                       method=algo,
                       tol=F_TOL,
                       options=options)

    t1 = time.clock()

    print 'Algorithm: ', algo
    print 'Total Time: ', t1
    print 'Number evals: ', res.nfev
    print 'F-cost: ', res.fun
