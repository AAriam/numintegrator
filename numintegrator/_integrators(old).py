import numpy as np

def integrate(method, f, y, x, dx, dx_last, tolerance):
    integrator = globals()[method.value]
    if method in Integrators.Explicit:
        integrate_explicit(integrator, f, y, x, dx_gen)
    elif method in Integrators.Implicit:
        integrate_implicit(integrator, f, y, x, dx_gen, tolerance)


def integrate_explicit(integrator, f, y, x, dx_gen):
    for n, _ in enumerate(x[1:]):
        dx = dx_gen.send(None)
        y_n = y[n]
        x_n = x[n]
        x_n1 = x[n + 1]
        y[n + 1][:] = integrator(f, dx, y_n, x_n, x_n1)
    return


def integrate_implicit(integrator, f, y, x, dx_gen, tolerance):
    for n, _ in enumerate(x[1:]):
        dx = dx_gen.send(None)
        y_n = y[n]
        x_n = x[n]
        x_n1 = x[n + 1]
        y_n1 = euler_explicit(f, dx, y_n, x_n, x_n1)
        y[n + 1][:] = fixed_point_iter(
            integrator, tolerance, f=f, dx=dx, y_n=y_n, x_n=x_n, x_n1=x_n1, y_n1=y_n1)
    return


class FirstOrder:

    class Explicit:

        @staticmethod
        def euler(f, dx, y_n, x_n, x_n1=None):
            f1 = f(x_n, y_n)
            y_n1 = y_n + dx * f1
            return y_n1

        @staticmethod
        def heun(f, dx, y_n, x_n, x_n1):
            f1 = f(x_n, y_n)
            f2 = f(x_n1, y_n + dx * f1)
            y_n1 = y_n + dx * (f1 + f2) / 2
            return y_n1

        @staticmethod
        def runge_kutta_order4(f, dx, y_n, x_n, x_n1=None):
            f1 = f(x_n, y_n)
            f2 = f(x_n + dx / 2, y_n + dx * f1 / 2)
            f3 = f(x_n + dx / 2, y_n + dx * f2 / 2)
            f4 = f(x_n + dx, y_n + dx * f3)
            y_n1 = y_n + dx * (f1 + 2 * f2 + 2 * f3 + f4) / 6
            return y_n1

    class Implicit:
        @staticmethod
        def fixed_point_iter(integrator, tolerance, **kwargs):
            y_n1_old = kwargs["y_n1"]
            y_n1_new = integrator(**kwargs)
            while (abs(y_n1_new - kwargs["y_n1"]) * 100 / kwargs["y_n1"]).mean() > tolerance:
                kwargs["y_n1"] = y_n1_new
                y_n1_new = integrator(**kwargs)
            return y_n1_new

        @staticmethod
        def euler_implicit(f, dx, y_n, y_n1, x_n1, x_n):
            f1 = f(x_n1, y_n1)
            y_n1_new = y_n + dx * f1
            return y_n1_new

        @staticmethod
        def crank_nicolson(f, dx, y_n, x_n, y_n1, x_n1):
            f1 = f(x_n, y_n)
            f2 = f(x_n1, y_n1)
            y_new = y_n + dx * (f1 + f2) / 2
            return y_new

        @staticmethod
        def midpoint_rule(f, dx, y_n, x_n, y_n1, x_n1):
            f1 = f((x_n + x_n1) / 2, (y_n + y_n1) / 2)
            y_new = y_n + dx * f1
            return y_new


class SecondOrder:

    @staticmethod
    def euler_explicit_2nd_order_ode(f, dx, y_n, x_n, y1_n):
        f1 = f(x_n, y_n)
        y1_n1 = y1_n + dx * f
        y_n1 = y_n + dx * y1_n + dx * dx * f1 / 2
        return np.array([y_n1, y1_n1])

    @staticmethod
    def verlet_2nd_order_ode():
        pass
