import numpy as np
from utility.interpolation import LinearInterpolator

# All penalty function names
GLOBAL_DIR_PENALTY = 'global_dir'
LAST_DIR_PENALTY = 'last_dir'
L2_WEIGHTS = 'l2_weights'
L1_WEIGHTS = 'l1_weights'
L2_CURVATURE = 'l2_curvature'
L1_CURVATURE = 'l1_curvature'
L2_DK = 'l2_dk'
L1_DK = 'l1_dk'
TARGET_POINTS_PENALTY = 'target_points_penalty'

def cosine_similarity(a, b):
    '''
    Cosine similarity function bounded between [0,1]

    :param a: vector a
    :param b: vector b
    :return: cosine similarity scaled between [0,1]
    '''
    dot_prod = np.sum(a * b)
    return (dot_prod / (np.linalg.norm(a) * np.linalg.norm(b)) + 1) / 2.0


def global_dir_penalty_func(psi, veh_model, args):
    '''
    Adds a penalty for every point calculated that is not in the direction of the args
    argument 'global_desired_theta'. It basically gets the path to go in one direction.

    :param psi: Curvature function
    :param veh_model: Vehicle model
    :param args: Penalty args
    :return: cost
    '''
    ks, dists = psi.eval_over_length(args['desired_spacing'])
    x_s = veh_model.simulate_over_length(ks, np.diff(dists))

    return 0.5 * np.sum(np.square((x_s[2, :] - args['global_desired_theta'])))


def last_dir_penalty_func(psi, veh_model, args):
    '''
    Adds a penalty for kust the last point that is not in the direction of the args
    argument 'last_desired_theta'. It basically gets the path to go in one direction.

    :param psi: Curvature function
    :param veh_model: Vehicle model
    :param args: Penalty args
    :return: cost
    '''
    ks, dists = psi.eval_over_length(args['desired_spacing'])
    x_s = veh_model.simulate_over_length(ks, np.diff(dists))

    return 0.5 * np.sum(np.square((x_s[2, -1] - args['last_desired_theta'])))


def l2_penalty_weights(psi, veh_model, args):
    '''
    L2 penalty on the weights of psi

    :param psi: Curvature function
    :param veh_model: Vehicle model
    :param args: Penalty args
    :return: cost
    '''
    return 0.5 * np.sum(np.square(psi.weights)) / psi.weights.size


def l1_penalty_weights(psi, veh_model, args):
    '''
    L1 penalty on the weights of psi

    :param psi: Curvature function
    :param veh_model: Vehicle model
    :param args: Penalty args
    :return: cost
    '''

    return np.sum(np.abs(psi.weights)) / psi.weights.size


def l2_penalty_curvature(psi, veh_model, args):
    '''
    Adds a penalty on the l2 of the curvature

    :param psi: Curvature function
    :param veh_model: Vehicle model
    :param args: Penalty args
    :return: cost
    '''
    ks, dists = psi.eval_over_length(args['desired_spacing'])
    return 0.5 * np.sum(np.square(ks))


def l1_penalty_curvature(psi, veh_model, args):
    '''
    Adds a penalty on the l1 of the curvature

    :param psi: Curvature function
    :param veh_model: Vehicle model
    :param args: Penalty args
    :return: cost
    '''
    ks, dists = psi.eval_over_length(args['desired_spacing'])
    return np.sum(np.abs(ks))


def l2_penalty_dk(psi, veh_model, args):
    '''
    Adds a penalty on the l2 of the rate of change of curvature

    :param psi: Curvature function
    :param veh_model: Vehicle model
    :param args: Penalty args
    :return: cost
    '''
    dks, dists = psi.eval_derv_over_length(args['desired_spacing'])
    return 0.5 * np.sum(np.square(dks))


def l1_penalty_dk(psi, veh_model, args):
    '''
    Adds a penalty on the l1 of the rate of change of curvature

    :param psi: Curvature function
    :param veh_model: Vehicle model
    :param args: Penalty args
    :return: cost
    '''
    dks, dists = psi.eval_derv_over_length(args['desired_spacing'])
    return np.sum(np.abs(dks))


def target_points_penalty(psi, veh_model, args):
    '''
    Adds a penalty for min distance to target points.

    Target points are of the form [distance, x, y]

    :param psi: Curvature function
    :param veh_model: Vehicle model
    :param args: Penalty args
    :return: cost
    '''
    t_points = args['target_points']

    ks, dists = psi.eval_over_length(args['desired_spacing'])
    x_s = veh_model.simulate_over_length(ks, np.diff(dists))

    cost = 0
    for t_p in t_points:
        point_dists = np.sum(np.square(x_s[:2, :].transpose() - t_p), axis=1)

        cost += point_dists.min()

    return 0.5 * cost / t_points.shape[0]


