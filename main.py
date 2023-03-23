import numpy as np
import matplotlib.pyplot as plt

class SeismicSimulationFDM:
    def __init__(self, nx, nt, dx, dt, vp, f0, src_x, rec_x):
        """
        Initializes the SeismicSimulationFDM class with the following parameters:

        :param nx: (int) Number of grid points in the x direction
        :param nt: (int) Number of time steps
        :param dx: (float) Spatial step size
        :param dt: (float) Time step size
        :param vp: (float) Velocity of the wave
        :param f0: (float) Dominant frequency of the Ricker wavelet
        :param src_x: (float) x-coordinate of the source location
        :param rec_x: (float) x-coordinate of the receiver location
        """
        self.nx = nx
        self.nt = nt
        self.dx = dx
        self.dt = dt
        self.vp = vp
        self.f0 = f0
        self.src_x = src_x
        self.rec_x = rec_x

    def ricker_wavelet(self, t):
        """
                Generates the Ricker wavelet time function.

                :param t: (numpy.ndarray) Array of time values
                :return: (numpy.ndarray) Array of amplitudes corresponding to the Ricker wavelet at the given time values
        """
        a = np.pi * self.f0 * (t - 1 / self.f0)
        return (1 - 2 * a**2) * np.exp(-a**2)

    def simulate(self):
        """
                Simulates the seismic wave propagation.

                :return: Tuple containing the displacement field and receiver data
        """
        u = np.zeros((self.nx, self.nt))
        v = np.zeros((self.nx, self.nt))
        a = np.zeros((self.nx, self.nt))

        src_idx = int(self.src_x / self.dx)
        rec_idx = int(self.rec_x / self.dx)

        src_time_function = self.ricker_wavelet(np.arange(self.nt) * self.dt)

        for t in range(1, self.nt):
            a[src_idx, t] = src_time_function[t]
            a[1:-1, t] += (u[2:, t-1] - 2*u[1:-1, t-1] + u[:-2, t-1]) / self.dx**2 * self.vp**2
            v[:, t] = v[:, t-1] + 0.5 * self.dt * (a[:, t] + a[:, t-1])
            u[:, t] = u[:, t-1] + self.dt * v[:, t]

        rec_data = u[rec_idx, :]

        return u, rec_data

    def visualize(self, u):
        plt.imshow(u, aspect='auto', cmap='seismic', extent=[0, self.nx * self.dx, self.nt * self.dt, 0], vmin=-0.1,
                   vmax=0.1)

        plt.xlabel('Distance (m)', color='blue', fontsize=14)
        plt.ylabel('Time (s)', color='blue', fontsize=14)
        plt.title('Seismic Simulation (FDM)', fontsize=14)
        plt.colorbar()

        ax = plt.gca()
        ax.tick_params(axis='x', colors='blue')
        ax.tick_params(axis='y', colors='blue')

        plt.savefig('seismic_simulation.png', dpi=300, bbox_inches='tight')
        plt.show()


if __name__ == '__main__':
    nx = 200
    nt = 1000
    dx = 10
    dt = 0.001
    vp = 2000
    f0 = 25
    src_x = 1000
    rec_x = 1500

    simulation = SeismicSimulationFDM(nx, nt, dx, dt, vp, f0, src_x, rec_x)
    u, rec_data = simulation.simulate()
    simulation.visualize(u)