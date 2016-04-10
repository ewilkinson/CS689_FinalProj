import numpy as np


class CurvatureExp(object):
    def __init__(self, N, L, sigint, zero_weights=False):
        self.N = N
        self.L = L

        self.shape = (self.N,)
        self.mus = np.linspace(start=0, stop=self.L, endpoint=True, num=self.N, dtype=np.float64)
        self.sigmas = np.ones(self.shape, dtype=np.float64) * sigint
        self.weights = np.random.uniform(low=-0.05, high=0.05, size=self.N)

        if zero_weights:
            self.weights = np.zeros(self.weights.shape, dtype=self.weights.dtype)

    def eval(self, x):
        return np.sum(self.weights * np.exp(-np.square(x - self.mus) / (2 * np.square(self.sigmas))))

    def eval_derv(self, x):
        sig_square = np.square(self.sigmas)
        diff_mu = x - self.mus
        return np.sum(self.weights * -diff_mu / (2 * sig_square) * np.exp(-np.square(diff_mu) / (2 * sig_square)))

    def eval_over_length(self, desired_spacing = 0.5):
        num_points = int(self.L / desired_spacing)
        dists = np.linspace(0, self.L, num=num_points, endpoint=True)

        ks = []
        for dist in dists:
            ks.append(self.eval(dist))

        return np.asarray(ks), dists

    def eval_derv_over_length(self, desired_spacing = 0.5):
        num_points = int(self.L / desired_spacing)
        dists = np.linspace(0, self.L, num=num_points, endpoint=True)

        ks = []
        for dist in dists:
            ks.append(self.eval_derv(dist))

        return np.asarray(ks, dtype=np.float64), dists


if __name__ == '__main__':
    L = 50
    desired_spacing = 0.5

    N = 20
    sigint = 3.0

    psi = CurvatureExp(N, L, sigint)

    ks, dists = psi.eval_over_length(0.5)
    dks, dists = psi.eval_derv_over_length(0.5)

    import matplotlib.pyplot as plt
    plt.ion()


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 14), sharex=True)
    ax1.plot(dists, ks)
    ax1.set_title('Random Curvature Gauss Basis Function')
    ax1.set_xlabel('Distance')
    ax1.set_ylabel('Curvature')

    ax2.plot(dists, dks)
    ax2.set_title('Random Curvature Derivative')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')

    for ax in [ax1, ax2]:
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                         ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(20)

    from vehicle_model.vehicle_model import VehicleModel

    X = np.asarray([0, 0, 0])
    model = VehicleModel(X, 1.0, VehicleModel.EXACT)

    X_s = model.simulate_over_length(ks, np.diff(dists))

    plt.figure(figsize=(16, 16))
    plt.plot(X_s[0, :], X_s[1, :])
    plt.title('Path plot of curvature function')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.ylim([-25,25])
    plt.show()
