import numpy as np


def create_t_series(t0, dt=None, n_steps=None, tn=None):
    """
    Create x-series (e.g. time-series) data for the integrator
    from input specifications.

    Parameters
    ----------
    t0 : int/float
        Initial value of the series.
    dt : int/float or array-like
        Integration step size (optional).
    n_steps : int
        Number of integration steps (optional).
    tn : int/float
        Last value of t (optional).

    Returns
    -------
        2D-tuple of numpy.ndarray
        First array is the t-series, and
        second array is the dt-series.

    Notes
    -----
    From the three optional parameters for defining the number of steps
    and the step-size (i.e. `dt`, `n_steps` and `tn`), exactly two arguments
    should be provided. An exception is when `dt` is provided as an array (for variable step-size),
    in which case `n_steps` and `tn` should not be provided.
    """

    def create_array(t0, tn, n_steps):
        return np.linspace(t0, tn, n_steps + 1)

    def calculate_tn(t0, dt, n_steps):
        return t0 + (n_steps * dt)

    # Verify that between `dt`, `n_steps` and `tn`
    # two and only two parameters are provided
    sum_none = sum(map(lambda val: val is None, [dt, n_steps, tn]))
    if isinstance(dt, (list, np.ndarray)):
        if sum_none != 2:
            raise ValueError("When dx is provided as an array, `n_steps` and `tn` should not be provided.")
        else:
            dt_array = np.array(dt)
            if len(dt.shape) != 1:
                raise ValueError("`dt` array must be one dimensional.")
            tn = t0 + dt.sum()
            t = create_array(t0, tn, dt.shape[0])
            return t, dt_array

    elif sum_none != 1:
        raise ValueError(
            "2 out of 3 parameters `n_steps`, `tn` and `t0` should be provided."
        )
    # If n_steps is provided, verify that it is a positive integer
    elif (n_steps is not None) and (not isinstance(n_steps, int) or n_steps <= 0):
        raise ValueError("`n_steps` must be a positive integer value.")

    # Find out which parameter is not provided
    # and create the t-series accordingly
    elif dt is None:
        t = create_array(t0, tn, n_steps)
        dt = (tn - t0) / n_steps
        dt_array = np.full(n_steps, dt)
        return t, dt_array

    elif tn is None:
        tn = calculate_tn(t0, dt, n_steps)
        t = create_array(t0, tn, n_steps)
        dt_array = np.full(n_steps, dt)
        return t, dt_array

    # now the only possibility is that n_steps is None
    else:
        n_steps_ = (tn - t0) / dt
        n_steps = int(n_steps_)
        # If the calculated required number of steps is an integer
        if np.isclose(n_steps_, n_steps):
            t = create_array(t0, tn, n_steps)
            dt_array = np.full(n_steps, dt)
        else:
            # create t-array with integer number of n_steps (floor)
            tn_ = calculate_tn(t0, dt, n_steps)
            t = create_array(t0, tn_, n_steps)
            # append tn to the end of the array
            t = np.append(t, tn)
            # create dt_array for actual required number of steps
            dt_array = np.full(n_steps + 1, dt)
            # calculate dt for last step and substitute with last element of dt-array
            dt_last = tn - tn_
            dt_array[-1] = dt_last
    return t, dt_array




