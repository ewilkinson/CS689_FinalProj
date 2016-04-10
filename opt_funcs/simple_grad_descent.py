from scipy.optimize import OptimizeResult
import numpy as np


def simple_grad(fun, x0, args=(), maxfev=None, stepsize=0.01,
                maxiter=100, callback=None, **options):
    '''
    Simple gradient method taken from the scipy minimize documentation.

    :param fun:
    :param x0:
    :param args:
    :param maxfev:
    :param stepsize:
    :param maxiter:
    :param callback:
    :param options:
    :return:
    '''
    bestx = x0
    besty = fun(x0, *args)
    funcalls = 1
    niter = 0
    improved = True
    stop = False

    while improved and not stop and niter < maxiter:
        improved = False
        niter += 1
        for dim in range(np.size(x0)):
            for s in [bestx[dim] - stepsize, bestx[dim] + stepsize]:
                testx = np.copy(bestx)
                testx[dim] = s
                testy = fun(testx, *args)
                funcalls += 1

                if testy < besty:
                    besty = testy
                    bestx = testx
                    improved = True

            if callback is not None:
                callback(bestx)
            if maxfev is not None and funcalls >= maxfev:
                stop = True
                break

    return OptimizeResult(fun=besty, x=bestx, nit=niter,
                          nfev=funcalls, success=(niter > 1))
