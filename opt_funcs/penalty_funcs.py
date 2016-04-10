import numpy as np

# All penalty function names
GLOBAL_DIR_PENALTY = 'global_dir'
L2_WEIGHTS = 'l2_weights'
L1_WEIGHTS = 'l1_weights'
L2_CURVATURE = 'l2_curvature'
L1_CURVATURE = 'l1_curvature'
L2_DK = 'l2_dk'
L1_DK = 'l1_dk'


def cosine_similarity(a, b):
    '''
    Cosine similarity function bounded between [0,1]

    :param a: vector a
    :param b: vector b
    :return: cosine similarity scaled between [0,1]
    '''
    dot_prod = np.sum(a * b)
    return (dot_prod / (np.linalg.norm(a) * np.linalg.norm(b)) + 1) / 2.0


def dir_penalty_func(psi, veh_model, args):
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


def l2_penalty_weights(psi, veh_model, args):
    '''
    L2 penalty on the weights of psi

    :param psi: Curvature function
    :param veh_model: Vehicle model
    :param args: Penalty args
    :return: cost
    '''
    return 0.5 * np.sum(np.square(psi.weights))


def l1_penalty_weights(psi, veh_model, args):
    '''
    L1 penalty on the weights of psi

    :param psi: Curvature function
    :param veh_model: Vehicle model
    :param args: Penalty args
    :return: cost
    '''

    return np.sum(np.abs(psi.weights))


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
