# import sys
from enum import Enum
import inspect
import numpy as np
from .integrators import *
from . import helpers as helpers


class Integrators(Enum):
    ODE_1_EXPLICIT_EULER = {"ode_order": 1, "implicit": False, "func": euler_explicit}
    ODE_1_EXPLICIT_HEUN = {"ode_order": 1, "implicit": False, "func": heun_explicit}
    ODE_1_EXPLICIT_RUNGE_KUTTA_ORDER4 = {"ode_order": 1, "implicit": False, "func": runge_kutta_order4}

    ODE_1_IMPLICIT_EULER = {"ode_order": 1, "implicit": True, "func": euler_implicit}
    ODE_1_IMPLICIT_CRANK_NICOLSON = {"ode_order": 1, "implicit": True, "func": crank_nicolson}
    ODE_1_IMPLICIT_MIDPOINT = {"ode_order": 1, "implicit": True, "func": midpoint_rule}

    ODE_2_EXPLICIT_EULER = {"ode_order": 2, "implicit": False, "func": euler_explicit_ode2}
    ODE_2_EXPLICIT_VERLET = {"ode_order": 2, "implicit": False, "func": verlet_explicit_ode2}
    ODE_2_EXPLICIT_VELOCITY_VERLET = {"ode_order": 2, "implicit": False, "func": velocity_verlet_explicit_ode2}
    ODE_2_EXPLICIT_YOSHIDA_LEAPFROG_ORDER4 = {"ode_order": 2, "implicit": False, "func": leapfrog_yoshida_order4}


def integrate(integrator, f, x0, v0=None, t0=0, dt=None, n_steps=None, tn=None,
              explicit_integrator=Integrators.ODE_1_EXPLICIT_EULER, tolerance=None, **kwargs):
    """
    Solve (system of) 1st or 2nd-order ordinary differential equation
    using a numerical integrator.

    Parameters:
    -----------
    method : Enum
        Integration method for solving the ODE.
        Available methods can be viewed and inputted
        from `numintegrator.Integrators`.
    f : function
        The right-hand side function of the ODE to be solved.
        The ODE should be in one of the following forms:
        dx(t)/dt = f(x(t))
        dx(t)/dt = f(x(t), t)
        dx(t)/dt = f(x(t), t, constant1, constant2, ...)
        d^2x(t)/dt^2 = f(x(t))
        d^2x(t)/dt^2 = f(x(t), t)
        d^2x(t)/dt^2 = f(x(t), t, constant1, constant2, ...)
        In any case, the first two arguments of the function
        must be reserved for x(t) and t, respectively, even when
        the function is actually only a function of one of them.
        All other arguments, if any, must be constants that are
        either provided as non-default arguments in the function itself,
        or as keyword arguments to this function (i.e. `integrate`).
    x0 : array-like
        Initial values of x, i.e. x(t0).
    v0 : array-like
        Initial values of v, i.e. dx(t0)/dt.
        Optional, i.e. only required for integration methods of 2nd-order ODEs.
    t0 : int/float
        Initial value of t (optional; default: 0).
    dt : int/float or array-like
        Integration step size (optional).
    n_steps : int
        Number of integration steps (optional).
    tn : int/float
        Last value of x (optional).
    explicit_integrator : Enum
        Explicit integrator used by the specified implicit method.
        Optional, i.e. only required for implicit integration methods.
    tolerance : float
        Optional; i.e. only required for implicit methods.
        The relative difference (in percent) between two
        fixed-point iterations (i.e. (x(n+1)_new - x(n+1)_old) * 100 / x(n+1)_old),
        which must be reached in order for the iteration to stop.
    kwargs : keyword arguments
        Non-default arguments (other than `x` and `t`) for the input function `f`.

    Returns
    -------
        tuple
        2D-tuple of (x, t) for 1st-order ODEs, or
        3D-tuple of (x, v, t) for 2nd-order ODEs,
        where each tuple element is a numpy.ndarray.

    Notes
    -----
    For better performance, the input function should have
    two parameters x(t) and t, even if it's only a function
    of x(t).

    From the three optional parameters for defining the number of steps
    and the step-size (i.e. `dt`, `n_steps` and `tn`), exactly two arguments
    should be provided. An exception is when `dt` is provided as an array (for variable step-size),
    in which case `n_steps` and `tn` should not be provided.

    Each returned array also contains its corresponding initial value.
    The length of each array is thus equal to `n_steps + 1`. E.g.:
    x = [x0, x1, x2, ..., x(n_steps)]
    """

    t, dt = helpers.create_t_series(t0, dt, n_steps, tn)
    # Create x-series and add the initial value
    x0 = np.array(x0)
    x = np.zeros((len(t), *x0.shape))
    x[0][...] = x0

    # Verify that the input function has at least two non-default arguments,
    # and if it has more, verify that they are provided as kwargs
    f_sig = inspect.signature(f)
    f_non_default_args = [
        name for name, value in f_sig.parameters.items()
        if value.default is inspect.Parameter.empty
    ]
    if len(f_non_default_args) < 2:
        raise ValueError("Input function must accept at least two non-default arguments.")
    elif len(f_non_default_args) > 2:
        for arg in f_non_default_args[2:]:
            if arg not in kwargs:
                raise ValueError(f"Non-default argument {arg} in the input function not provided.")
    else:
        # the function has exactly two non-default arguments
        pass

    # Verify that the input integration method exists:
    if integrator not in Integrators:
        raise ValueError(f"Integration method `{integrator}` not recognized.")

    # If the integration method is for 2nd-order ODEs,
    elif integrator.value["ode_order"] == 2:
        # verify that v0 is provided:
        if v0 is None:
            raise ValueError("For integrators of 2nd-order ODEs `v0` should be provided.")
        # verify that v0 and x0 have the same shape:
        else:
            v0 = np.array(v0)
            v = np.zeros((len(t), *v0.shape))
            v[0][...] = v0
            if v.shape != x.shape:
                raise ValueError("`x0` and `v0` should have the same shape.")
            # everything is okay; call the integrator
            else:
                integrator.value["func"](f, x, v, t, dt, **kwargs)
                return x, v, t

    # If the integration method is implicit (for 1st-order ODE),
    elif integrator.value["implicit"]:
        # verify that a second explicit method is provided
        if explicit_integrator not in Integrators or explicit_integrator.value["implicit"]:
            raise ValueError("Implicit integrator requires a second explicit integrator to be provided.")
            # verify that `tolerance` is provided:
        elif tolerance is None:
            raise ValueError("For implicit integrators, `tolerance` should be provided.")
        # everything is okay; call the integrator
        else:
            integrator.value["func"](f, x, t, dt, explicit_integrator.value["func"], tolerance, **kwargs)

    # The integrator should then be an explicit method for 1st-order ODE
    else:
        integrator.value["func"](f, x, t, dt, **kwargs)

    return x, t


if __name__ == "__main__":
    # for arg in sys.argv:
    #     print(arg, type(arg))

    # solver.solve()
    pass
