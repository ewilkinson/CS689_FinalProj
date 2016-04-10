import numpy as np
from utility.interpolation import LinearInterpolator

class ExactTracker(object):
    def __init__(self, L):
        self.L = L

    def simulate(self, X, k_path, dist_along_path):
        ds = np.diff(dist_along_path)

        theta_s = np.zeros(k_path.shape)
        x_s = np.zeros(k_path.shape)
        y_s = np.zeros(k_path.shape)

        x_s[0] = X[0]
        y_s[0] = X[1]
        theta_s[0] = X[2]

        for i in range(len(ds)):
            theta_s[i + 1] = theta_s[i] + self.L * k_path[i] * ds[i]
            x_s[i + 1] = x_s[i] + np.cos(theta_s[i]) * ds[i]
            y_s[i + 1] = y_s[i] + np.sin(theta_s[i]) * ds[i]

        for i in range(len(ds)):
            x_s[i + 1] = x_s[i] + np.cos(theta_s[i]) * ds[i]
            y_s[i + 1] = y_s[i] + np.sin(theta_s[i]) * ds[i]

        return np.delete(np.vstack((x_s, y_s, theta_s)), 0, axis=1)


class VehicleModel(object):
    EXACT = "exact"

    def __init__(self, x0, L, model_type):
        assert type(x0) == np.ndarray
        assert type(model_type) == str


        # Let's just give the vehicle a constant velocity
        self.vel = 1.0
        self.x = x0

        if model_type == VehicleModel.EXACT:
            self.model = ExactTracker(L)
        else:
            raise RuntimeError("Passed a model type not recognized")

    def simulate_over_length(self, k_path, k_spacing):
        dist_along_path = np.cumsum(k_spacing)
        dist_along_path = np.insert(dist_along_path, 0, 0)

        x_sim = self.model.simulate(self.x, k_path, dist_along_path)

        x_sim = np.hstack((self.x.reshape(3, 1), x_sim))
        return x_sim

    def simulate(self, k_path, k_spacing, time):
        assert type(k_path) == np.ndarray
        assert type(k_spacing) == np.ndarray
        assert len(k_path) == len(k_spacing) + 1
        assert time > 0

        dist_to_simulate = self.vel * time
        dist_along_path = np.cumsum(k_spacing)
        dist_along_path = np.insert(dist_along_path, 0, 0)

        assert dist_along_path.shape == k_path.shape

        if dist_to_simulate > dist_along_path[-1]:
            print dist_to_simulate, dist_along_path[-1]
            raise RuntimeError("Asked to simulate more than local path provided.")

        for i, dist in enumerate(dist_along_path):
            if dist > dist_to_simulate:
                # we know how far foward to simulate. Truncate the input arrays
                # gamma is how far along the path to interpolate
                k_path_trunc = k_path[:i]
                dist_along_path_trunc = dist_along_path[:i]

                k_path_interp = LinearInterpolator(k_path, dist_along_path)

                np.append(k_path_trunc, k_path_interp(dist))
                np.append(dist_along_path_trunc, dist)

                break

        v_path = np.ones(k_path_trunc.shape) * self.vel

        x_sim = self.model.simulate(self.x, k_path_trunc, dist_along_path_trunc)
        x_sim = np.hstack((self.x.reshape(3, 1), x_sim))

        return x_sim


if __name__ == '__main__':
    x = np.arange(10)
    x_dists = np.arange(10) * 0.1
    x_interp = LinearInterpolator(x, x_dists)

    print 'Interpolation at 0.3 : ', x_interp(0.3)
    print 'Interpolation at 0.35 : ', x_interp(0.35)
    print 'Interpolation at 0 : ', x_interp(0.0)
    print 'Interpolation at 100 : ', x_interp(100)

    N = 50
    dist = np.pi * 2.0
    X = np.asarray([0, 0, 0])

    k_path = np.sin(np.linspace(0, np.pi * 4, num=N))
    k_spacing = np.ones(shape=(N - 1,), dtype=np.float64) * dist / (N - 1)
    dist_along_path = np.cumsum(k_spacing)
    dist_along_path = np.insert(dist_along_path, 0, 0)

    # tracker = ExactTracker(1.0)
    # X_s = tracker.simulate(X, k_path, dist_along_path, None, None)

    model = VehicleModel(X, 1.0, VehicleModel.EXACT)

    X_s = model.simulate(k_path, k_spacing, 5.0)
    import matplotlib.pyplot as plt

    plt.ion()

    plt.figure(figsize=(16, 16))
    plt.plot(X_s[0, :], X_s[1, :])
    plt.title('Path plot of curvature function')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    plt.figure(figsize=(16, 16))
    plt.plot(dist_along_path, k_path)
    plt.title('Curvature command')
    plt.xlabel('Distance')
    plt.ylabel('Curvature')
    plt.show()

    model.x = X
    Y_s = model.simulate_over_length(k_path, k_spacing)
    import matplotlib.pyplot as plt

    plt.ion()

    plt.figure(figsize=(16, 16))
    plt.plot(Y_s[0, :], Y_s[1, :])
    plt.title('Path plot of curvature function')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    plt.figure(figsize=(16, 16))
    plt.plot(dist_along_path, k_path)
    plt.title('Curvature command')
    plt.xlabel('Distance')
    plt.ylabel('Curvature')
    plt.show()
