import numpy as np


class CurvatureExp(object):
    def __init__(self, N, L, sigint):
        self.N = N
        self.L = L

        self.shape = (self.N,)
        self.mus = np.linspace(start=0, stop=self.L, endpoint=True, num=self.N, dtype=np.float64)
        self.sigmas = np.ones(self.shape, dtype=np.float64) * sigint
        self.weights = np.random.uniform(low=-0.05, high=0.05, size=self.N)

    def eval(self, x):
        return np.sum(self.weights * np.exp(-np.square(x - self.mus) / (2 * np.square(self.sigmas))))

    def eval_derv(self, x):
        pass

    def eval_over_length(self, desired_spacing = 0.5):
        num_points = int(self.L / desired_spacing)
        dists = np.linspace(0, self.L, num=num_points, endpoint=True)

        ks = []
        for dist in dists:
            ks.append(self.eval(dist))

        return np.asarray(ks), dists

if __name__ == '__main__':
    L = 50
    desired_spacing = 0.5

    N = 20
    sigint = 1.5

    psi = CurvatureExp(N, L, sigint)

    ks, dists = psi.eval_over_length(0.5)

    import matplotlib.pyplot as plt
    plt.ion()

    ks = np.asarray(ks, dtype=np.float64)
    plt.figure(figsize=(15, 15))
    plt.plot(dists, ks)
    plt.title('Random Curvature Gauss Basis Function')
    plt.xlabel('Distance')
    plt.ylabel('Curvature')
    plt.show()

    from vehicle_model.vehicle_model import VehicleModel

    X = np.asarray([0, 0, 0])
    model = VehicleModel(X, 1.0, VehicleModel.EXACT)

    X_s = model.simulate_over_length(ks, np.diff(dists))

    plt.figure(figsize=(16, 16))
    plt.plot(X_s[0, :], X_s[1, :])
    plt.title('Path plot of curvature function')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
