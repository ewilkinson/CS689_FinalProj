import numpy as np
import matplotlib.pyplot as plt
from opt_funcs.penalty_funcs import target_points_penalty


def plot_weight_costs(psi, veh_model, penalty_args, test_idxes=[0, 10, 20]):
    '''
    Plots the costs for the target_points_penalty function at the provided weight params in
    test_indxes

    :param psi: The curvature function
    :param veh_model: The vehicle model
    :param penalty_args: The optimization penalty arguments
    :param test_idxes: Indices into the psi.weights vector whose cost should be examined.
    :return:
    '''
    costs = []
    test_weights = []
    orig_weights = np.copy(psi.weights)
    for test_idx in test_idxes:
        psi.weights = np.copy(orig_weights)
        w0 = orig_weights[test_idx]
        w0s = w0 + np.linspace(-1.57, 1.57, num=100, endpoint=True, dtype=np.float64)
        test_weights.append(w0s)

        costs.append([])
        for w in w0s:
            psi.weights[test_idx] = w
            costs[-1].append(target_points_penalty(psi, veh_model, penalty_args))

    costs = np.asarray(costs)
    test_weights = np.asarray(test_weights)
    psi.weights = np.copy(orig_weights)

    fig, (ax1) = plt.subplots(1, 1, figsize=(14, 14))

    for ws, cost, test_idx in zip(test_weights, costs, test_idxes):
        ax1.plot(ws, cost, label='Weight index: ' + str(test_idx))

    ax1.set_title('Target Cost Function vs. Weights at Solution')
    ax1.set_xlabel('Weight Value')
    ax1.set_ylabel('Cost')
    ax1.legend()

    for ax in [ax1]:
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                         ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(20)

    plt.show()
