from scipy.optimize import OptimizeResult
from opt_funcs.adagrad import func_derv
import numpy as np


def adadelta(fun, x0, args=(), maxfev=None, rho_dx=0.8, rho_g=0.8,
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

    has_init = False
    rms_grad = np.zeros(shape=x0.shape, dtype=np.float64)

    improvement_tolerance = options['tol']
    MAX_IMPROVE_FAILS = 50
    improvement_fail_counter = 0

    while niter < maxiter:
        niter += 1

        grad = func_derv(fun, x, args)
        funcalls += np.size(x) * 2

        if not has_init:
            rms_grad = np.square(grad)
            rms_dx = 0
        else:
            rms_grad = rho_g * rms_grad + (1.0 - rho_g) * np.square(grad)

        dx = - np.sqrt(rms_dx + 1e-7) / (np.sqrt(rms_grad + 1e-7)) * grad
        x += dx

        rms_dx = rho_dx * rms_dx + (1.0 - rho_dx) * np.square(dx)

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
            callback(x)
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

    res = minimize(rosen, x0, method=adadelta,
                   tol=1e-4,
                   options={'maxiter': 10000, 'eta': 0.01})

    print 'ADADELTA RESULT'
    print 'Success? :', res.success
    print 'Optimal Value: ', res.x
    print 'Optimal f-val: ', res.fun
    print 'Iterations: ', res.nit
    print 'Func calls: ', res.nfev
