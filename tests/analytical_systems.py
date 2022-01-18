import numpy as np


class Harmonic_oscillator_1d_single_particle:
    """
    d^2/dt^2 x(t) = -kx(t)/m
    """

    def __init__(self, k=1, m=1):
        self.k = k
        self.m = m

    def analytical_solution(self, x_0, v_0, t):
        w = np.sqrt(self.k / self.m)
        phi = np.arctan(x_0 * w / v_0) - w * t[0]
        if x_0 != 0:
            A = x_0 / np.sin(w * t[0] + phi)
        else:
            A = v_0 / (w * np.cos(w * t[0] + phi))
        self.x_analytical = A * np.sin(w * t + phi)
        v = A * w * np.cos(w * t + phi)
        return x, v

    def f_ode2(self, x, t):
        return -self.k * x / self.m

    def f_ode1(self, x_v, t):
        f_x = x_v[1]
        f_v = -self.k * x_v[0] / self.m
        return np.array([f_x, f_v])


class ode1:
    def analytic_sol(self, t):
        return t + np.sqrt(1 + 2 * t ** 2)

    def f(self, x, t):
        return (x + t) / (x - t)
