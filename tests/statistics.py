import numpy as np


def relative_error(x_numerical, x_analytical):
    return np.abs(x_analytical - x_numerical) / x_numerical
