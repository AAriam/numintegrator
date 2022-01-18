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


def integrate_implicit(integrator, f, x, t, dt_gen, tolerance):
    for n, _ in enumerate(t[1:]):
        dt = dt_gen.send(None)
        x_n = x[n]
        t_n = t[n]
        t_n1 = t[n + 1]
        x_n1 = euler_explicit(f, dt, x_n, t_n, t_n1)
        x[n + 1][:] = fixed_point_iter(
            integrator, tolerance, f=f, dt=dt, x_n=x_n, t_n=t_n, x_n1=x_n1
        )
    return


class FirstOrder:
    class Implicit:
        @staticmethod
        def fixed_point_iter(integrator, tolerance, **kwargs):
            x_n1_old = kwargs["x_n1"]
            x_n1_new = integrator(**kwargs)
            while (
                abs(x_n1_new - kwargs["x_n1"]) * 100 / kwargs["x_n1"]
            ).mean() > tolerance:
                kwargs["x_n1"] = x_n1_new
                x_n1_new = integrator(**kwargs)
            return x_n1_new
