import numpy as np


class CurvatureExp(object):
    def __init__(self, N, L, sigint):
        self.N = N
        self.L = L

        self.shape = (self.N,)
        self.mus = np.linspace(start=0, stop=self.L, endpoint=True, num=self.N, dtype=np.float64)
        self.sigmas = np.ones(self.shape, dtype=np.float64) * sigint
        self.weights = np.random.uniform(low=-0.1, high=0.1, size=self.N)

    def eval(self, x):
        return np.sum(self.weights * np.exp(-np.square(x - self.mus) / (2 * np.square(self.sigmas))))


if __name__ == '__main__':
    L = 10
    desired_spacing = 0.5

    N = int(L / desired_spacing) + 1

    psi = CurvatureExp(N, L, desired_spacing / 2.0)

    dists = np.linspace(0, L, num=1e3, endpoint=True)

    import matplotlib.pyplot as plt

    ks = []
    for dist in dists:
        ks.append(psi.eval(dist))

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

    model.simulate(ks, np.diff(dists), 5.0)
    import matplotlib.pyplot as plt

    plt.ion()

    X_s = model.x_s
    plt.figure(figsize=(16, 16))
    plt.plot(X_s[0, :], X_s[1, :])
    plt.title('Path plot of curvature function')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
