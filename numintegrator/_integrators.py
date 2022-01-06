import numpy as np


def euler_explicit(f, x, t, dt, **kwargs):
    for n in range(len(dt)):
        f1 = f(x[n], t[n], **kwargs)
        x[n + 1][...] = x[n] + f1 * dt[n]
    return


def heun_explicit(f, x, t, dt, **kwargs):
    for n in range(len(dt)):
        f1 = f(x[n], t[n], **kwargs)
        f2 = f(x[n] + f1 * dt[n], t[n + 1], **kwargs)
        x[n + 1][...] = x[n] + (f1 + f2) * (dt[n] / 2)
    return


def runge_kutta_order4(f, x, t, dt, **kwargs):
    for n in range(len(dt)):
        f1 = f(x[n], t[n], **kwargs)
        f2 = f(x[n] + f1 * dt[n] / 2, t[n] + dt[n] / 2, **kwargs)
        f3 = f(x[n] + f2 * dt[n] / 2, t[n] + dt[n] / 2, **kwargs)
        f4 = f(x[n] + f3 * dt[n], t[n] + dt[n], **kwargs)
        x[n + 1][...] = x[n] + (f1 + 2 * f2 + 2 * f3 + f4) * dt[n] / 6
    return


def euler_implicit(f, x, t, dt, explicit_integrator, tolerance, **kwargs):
    for n in range(len(dt)):
        explicit_integrator(f, x[n:n + 2], t[n:n + 2], dt[n:n + 1], **kwargs)
        f1 = f(x[n + 1], t[n + 1], **kwargs)
        x_next_old = np.array(x[n + 1])
        x[n + 1][...] = x[n] + f1 * dt[n]
        while (abs(x[n + 1] - x_next_old) * 100 / x[n + 1]).mean() > tolerance:
            f1 = f(x[n + 1], t[n + 1], **kwargs)
            x_next_old[...] = x[n + 1]
            x[n + 1][...] = x[n] + f1 * dt[n]
    return


def crank_nicolson(f, x, t, dt, explicit_integrator, tolerance, **kwargs):
    for n in range(len(dt)):
        explicit_integrator(f, x[n:n + 2], t[n:n + 2], dt[n:n + 1], **kwargs)
        f1 = f(x[n], t[n], **kwargs)
        f2 = f(x[n + 1], t[n + 1], **kwargs)
        x_next_old = np.array(x[n + 1])
        x[n + 1][...] = x[n] + (f1 + f2) * dt[n] / 2
        while (abs(x[n + 1] - x_next_old) * 100 / x[n + 1]).mean() > tolerance:
            f1 = f(x[n], t[n], **kwargs)
            f2 = f(x[n + 1], t[n + 1], **kwargs)
            x_next_old[...] = x[n + 1]
            x[n + 1][...] = x[n] + (f1 + f2) * dt[n] / 2
    return


def midpoint_rule(f, x, t, dt, explicit_integrator, tolerance, **kwargs):
    for n in range(len(dt)):
        explicit_integrator(f, x[n:n + 2], t[n:n + 2], dt[n:n + 1], **kwargs)
        f1 = f((x[n] + x[n + 1]) / 2, (t[n] + t[n + 1]) / 2, **kwargs)
        x_next_old = np.array(x[n + 1])
        x[n + 1][...] = x[n] + f1 * dt[n]
        while (abs(x[n + 1] - x_next_old) * 100 / x[n + 1]).mean() > tolerance:
            f1 = f((x[n] + x[n + 1]) / 2, (t[n] + t[n + 1]) / 2, **kwargs)
            x_next_old[...] = x[n + 1]
            x[n + 1][...] = x[n] + f1 * dt[n]
    return


def euler_explicit_ode2(f, x, v, t, dt, **kwargs):
    for n in range(len(dt)):
        f1 = f(x[n], t[n], **kwargs)
        x[n + 1][...] = x[n] + v[n] * dt[n] + f1 * dt[n] * dt[n] / 2
        v[n + 1][...] = v[n] + f1 * dt[n]
    return


def verlet_explicit_ode2(f, x, v, t, dt, **kwargs):
    euler_explicit_ode2(f, x[:2], v[:2], t[:1], dt[:1], **kwargs)
    for n in range(1, len(dt)):
        f1 = f(x[n], t[n], **kwargs)
        x[n + 1][...] = 2 * x[n] - x[n - 1] + f1 * dt[n] * dt[n]
        v[n][...] = (x[n + 1] - x[n - 1]) / (dt[n] + dt[n - 1])
    # integrate an extra step to calculate the last dy
    f1 = f(x[-1], t[-1], **kwargs)
    y_n2 = 2 * x[-1] - x[-2] + f1 * dt[-1] * dt[-1]
    v[-1][...] = (y_n2 - x[-2]) / (dt[-1] * 2)
    return


def velocity_verlet_explicit_ode2(f, x, v, t, dt, **kwargs):
    f1 = f(x[0], t[0], **kwargs)
    for n in range(len(dt)):
        v_half_step = v[n] + f1 * dt[n] / 2
        x[n + 1][...] = x[n] + v_half_step * dt[n]
        f2 = f(x[n + 1], t[n + 1], **kwargs)
        v[n + 1][...] = v_half_step + f2 * dt[n] / 2
        f1[...] = f2
    return


def leapfrog_yoshida_order4(f, x, v, t, dt, **kwargs):
    w0 = -(2 ** (1 / 3)) / (2 - 2 ** (1/3))
    w1 = 1 / (2 - 2 ** (1/3))
    c1 = c4 = w1 / 2
    c2 = c3 = (w0 + w1) / 2
    d1 = d3 = w1
    d2 = w0
    for n in range(len(dt)):
        xn = x[n]
        vn = v[n]
        for c, d in [(c1, d1), (c2, d2), (c3, d3)]:
            xn += c * vn * dt[n]
            vn += d * f(xn, t[n], **kwargs) * dt[n]
        x[n + 1][...] = xn + c4 * vn * dt[n]
        v[n + 1][...] = vn
    return
