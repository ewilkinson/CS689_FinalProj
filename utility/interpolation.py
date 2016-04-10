import numpy as np


class LinearInterpolator(object):
    def __init__(self, x, x_dists):
        assert type(x) == np.ndarray
        assert type(x_dists) == np.ndarray
        assert x_dists.shape == x.shape

        self.x = x
        self.x_dists = x_dists

    def __call__(self, dist):
        for i, x_dist in enumerate(self.x_dists):
            if x_dist > dist:
                # we know how far foward to simulate. Truncate the input arrays
                # gamma is how far along the path to interpolate
                alpha = x_dist - dist
                beta = dist - self.x_dists[i - 1]
                gamma = alpha / (alpha + beta)

                x_interpolate = gamma * self.x[i - 1] + (1 - gamma) * self.x[i]
                return x_interpolate

        return self.x[-1]
