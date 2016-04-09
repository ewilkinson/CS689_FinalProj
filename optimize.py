import scipy.optimize as opt
import numpy as np
from curvature import CurvatureExp
from vehicle_model.vehicle_model import VehicleModel
from opt_funcs.penalty_funcs import *

# Create a vehicle at the origin 0,0,0
X = np.asarray([0, 0, 0])
veh_model = VehicleModel(X, 1.0, VehicleModel.EXACT)


# Let's see if we can get all points to line up in direction

global_desired_theta = 0.392


# Create a psi function. This integrates over length L, contains N gaussian kernels
# and has an initial sigma value of whatever is specified in sigint
L = 50
desired_spacing = 0.5
N = 50
sigint = 3.0
psi = CurvatureExp(N=N, L=L, sigint=sigint)


penalty_args = {'desired_spacing': desired_spacing,
                'global_desired_theta': global_desired_theta}

dir_alpha = 1.0
l1_weights_alpha = 1.0
l2_weights_alpha = 200.0

def min_func(x):
    # TODO eventually I will want to unpack the parameters differently if they start
    # changing mu and sigma values as well
    psi.weights = x
    dir_cost = dir_penalty_func(psi, veh_model, penalty_args)
    l1_weights_cost = l1_penalty_weights(psi, veh_model, penalty_args)

    total_cost = dir_alpha * dir_cost + l1_weights_alpha * l1_weights_cost

    return total_cost


x0 = np.copy(psi.weights)

# res = opt.minimize(min_func, x0, method='Nelder-Mead', tol=1e-6, options={'maxfev': 8000, 'maxiter': 8000})
res = opt.minimize(min_func, x0, method='L-BFGS-B', tol=1e-6, options={'disp': True})

print 'Optimization result : ', res.success
print 'Number of evaluations : ', res.nfev
print 'Minimum value of f : ', res.fun


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
