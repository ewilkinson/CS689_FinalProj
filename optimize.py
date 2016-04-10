import scipy.optimize as opt
import numpy as np
from curvature import CurvatureExp
from vehicle_model.vehicle_model import VehicleModel
from opt_funcs.penalty_funcs import *
from opt_funcs.cost_trace import CostTrace

# Create a vehicle at the origin 0,0,0
veh_model = VehicleModel(np.asarray([0, 0, 0]), 1.0, VehicleModel.EXACT)


# Create a psi function. This integrates over length L, contains N gaussian kernels
# and has an initial sigma value of whatever is specified in sigint
L = 50
desired_spacing = 0.5
N = 50
sigint = 3.0
initial_weights_zero = False
psi = CurvatureExp(N=N,
                   L=L,
                   sigint=sigint,
                   use_cache=True,
                   zero_weights=initial_weights_zero)


# Parameters related to optimization
penalty_args = {'desired_spacing': desired_spacing,
                'global_desired_theta': 0.392}

# 200 is a good value for l2 weights
penalty_weights = {GLOBAL_DIR_PENALTY: 1.0,
                   L1_WEIGHTS: 1.0,
                   L2_WEIGHTS: 0.0,
                   L1_CURVATURE: 0.0,
                   L2_CURVATURE: 0.0,
                   L2_DK: 100.0,
                   L1_DK: 0.0}
penalty_funcs = {GLOBAL_DIR_PENALTY: dir_penalty_func,
                 L1_WEIGHTS: l1_penalty_weights,
                 L2_WEIGHTS: l2_penalty_weights,
                 L1_CURVATURE: l1_penalty_curvature,
                 L2_CURVATURE: l2_penalty_curvature,
                 L2_DK: l2_penalty_dk,
                 L1_DK: l1_penalty_dk}


# check that every penalty weight has an associated penalty func
weight_set = set(penalty_weights.keys())
func_set = set(penalty_funcs.keys())
assert len(weight_set.intersection(func_set)) == len(weight_set)

cost_trace = CostTrace(penalty_weights, SAMPLE_RATE=100)


def min_func(x, penalty_weights, penalty_args, cost_trace):
    # TODO eventually I will want to unpack the parameters differently if they start
    # changing mu and sigma values as well
    psi.weights = x

    total_cost = 0

    # for each weight that is non zero evaluate the penalty and add to total cost
    for key in penalty_weights.keys():
        if penalty_weights[key] != 0:
            penalty_func = penalty_funcs[key]
            penalty_weight = penalty_weights[key]
            p_cost = penalty_weight * penalty_func(psi, veh_model, penalty_args)

            cost_trace.add_penalty(key, p_cost)

            total_cost += p_cost

    return total_cost


x0 = np.copy(psi.weights)

# res = opt.minimize(min_func, x0, method='Nelder-Mead', tol=1e-6, options={'maxfev': 8000, 'maxiter': 8000})
res = opt.minimize(min_func, x0, args=(penalty_weights, penalty_args, cost_trace), method='L-BFGS-B', tol=1e-6,
                   options={'disp': True})

print 'Optimization result : ', res.success
print 'Number of evaluations : ', res.nfev
print 'Minimum value of f : ', res.fun

cost_trace.plot_trace()

# PLOT THE RESULTS
import matplotlib.pyplot as plt

psi.weights = x0
ks, dists = psi.eval_over_length(desired_spacing)
Y_s = veh_model.simulate_over_length(ks, np.diff(dists))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 14))
ax1.plot(dists, ks)
ax1.set_title('Initial Curvature Function')
ax1.set_xlabel('Distance')
ax1.set_ylabel('Curvature')

ax2.plot(Y_s[0, :], Y_s[1, :])
ax2.set_title('Initial Path Plot')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_ylim([-25, 25])

for ax in [ax1, ax2]:
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(20)

plt.show()


# PLOT NEW SOLUTION X
psi.weights = res.x
ks, dists = psi.eval_over_length(desired_spacing)
Y_s = veh_model.simulate_over_length(ks, np.diff(dists))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 14))
ax1.plot(dists, ks)
ax1.set_title('Solution Curvature Function')
ax1.set_xlabel('Distance')
ax1.set_ylabel('Curvature')

ax2.plot(Y_s[0, :], Y_s[1, :])
ax2.set_title('Solution Path Plot')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_ylim([-25, 25])

for ax in [ax1, ax2]:
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(20)

plt.show()
