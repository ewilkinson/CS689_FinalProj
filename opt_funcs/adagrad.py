from scipy.optimize import OptimizeResult
import numpy as np


def func_derv(fun, x, args=(), epsilon=1e-8):
    grad = np.zeros(shape=x.shape[0], dtype=np.float64)

    x_copy = np.array(x, dtype=np.float64, copy=True)
    for dim in range(np.size(x)):
        val_copy = x_copy[dim]

        # evaluate the derivative
        x_copy[dim] = val_copy + epsilon
        g1 = fun(x_copy, *args)
        x_copy[dim] = val_copy - epsilon
        g2 = fun(x_copy, *args)

        # calc derv and reset x
        x_copy[dim] = val_copy
        grad[dim] = (g1 - g2) / (2 * epsilon)

    return grad


def adagrad(fun, x0, args=(), maxfev=None, eta=0.01,
            maxiter=10000, callback=None, **options):
    '''
    Implementation of ADAGRAD

    :return:
    '''
    x = np.array(x0, dtype=np.float64, copy=True)
    bestx = x0
    besty = fun(x0, *args)
    funcalls = 1
    niter = 0

    G_WINDOW_SIZE = 20
    sqr_g_window = np.zeros(shape=(G_WINDOW_SIZE,) + x0.shape, dtype=np.float64)
    g_cycle_idx = 0

    improvement_tolerance = options['tol']
    MAX_IMPROVE_FAILS = 50
    improvement_fail_counter = 0

    while niter < maxiter:
        niter += 1

        grad = func_derv(fun, x, args)
        funcalls += np.size(x) * 2

        sqr_g_window[g_cycle_idx, :] = np.square(grad)
        g_cycle_idx = (g_cycle_idx + 1) % G_WINDOW_SIZE

        sqrt_sum_square_g = np.sqrt(np.sum(sqr_g_window, axis=0))
        x += - eta / (sqrt_sum_square_g + 1e-7) * grad

        y_prime = fun(x, *args)
        funcalls += 1

        if besty - y_prime > improvement_tolerance:
            besty = y_prime
            bestx = x
            improvement_fail_counter = 0
        else:
            improvement_fail_counter += 1
            if improvement_fail_counter >= MAX_IMPROVE_FAILS:
                break

        if callback is not None:
            callback(bestx)
        if maxfev is not None and funcalls >= maxfev:
            break

    return OptimizeResult(fun=besty, x=bestx, nit=niter,
                          nfev=funcalls, success=(niter > 1))


if __name__ == '__main__':
    from scipy.optimize import minimize


    def rosen(x):
        """The Rosenbrock function"""
        return sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)


    def squared_func(x):
        return np.sum(np.square(x))


    x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
    res = minimize(rosen, x0, method='nelder-mead',
                   options={'xtol': 1e-8, 'disp': True})

    print 'Nelder-mead RESULT'
    print 'Success? :', res.success
    print 'Optimal Value: ', res.x
    print 'Optimal f-val: ', res.fun
    print 'Iterations: ', res.nit
    print 'Func calls: ', res.nfev

    res = minimize(rosen, x0, method=adagrad,
                   tol=1e-4,
                   options={'maxiter': 10000, 'eta': 0.01})

    print 'ADAGRAD RESULT'
    print 'Success? :', res.success
    print 'Optimal Value: ', res.x
    print 'Optimal f-val: ', res.fun
    print 'Iterations: ', res.nit
    print 'Func calls: ', res.nfev
