#  PyMODAlib, a Python implementation of the algorithms from MODA (Multiscale Oscillatory Dynamics Analysis).
#  Copyright (C) 2020 Lancaster University
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
Basic signal operations, such as pre-processing and down-sampling.
"""

import warnings
from typing import Optional

import numpy as np
from numpy import ndarray

from pymodalib.implementations.python.signals.preprocessing import preprocess_impl


class ResamplingException(Exception):
    """ """


def resampl_flow(
    signal: ndarray, original_freq: float, resample_freq: float
) -> ndarray:
    """
    Down-samples a signal, using a moving-average.

    Parameters
    ----------
    signal : ndarray
        [1D array] The signal to down-sample.
    original_freq : float
        The original sampling frequency of the signal.
    resample_freq : float
        The sampling frequency of the down-sampled signal.

    Returns
    -------
    ndarray
        [1D array] The down-sampled signal.
    """
    try:
        x, y = signal.shape

        if x > 1 and y > 1:
            raise ResamplingException(
                f"Signal dimensions are not valid for this function: ({x}, {y})."
            )
        elif x == 1:
            signal = signal.T
    except ValueError:
        pass

    L = len(signal)
    ratio = original_freq / resample_freq
    if int(ratio) != ratio:
        warnings.warn(
            f"The ratio between original and resampled sampling frequencies is {ratio}, not an integer. "
        )

    output = np.empty(np.int(np.floor(L / ratio)))
    ratio = int(ratio)

    for j in range(0, len(output)):
        start = np.int(np.ceil(j * ratio))
        output[j] = np.mean(signal[start : start + ratio])

    return output


def preprocess(
    signal: ndarray,
    fs: float,
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
) -> ndarray:
    """
    Pre-processes a signal, performing filtering and de-trending.

    Parameters
    ----------
    signal : ndarray
        [1D array] The signal to pre-process.
    fs : float
        The sampling frequency of the signal.
    fmin : float, optional
        The minimum frequency for filtering.
    fmax : float, optional
        The maximum frequency for filtering.

    Returns
    -------
    ndarray
        [1D array] The pre-processed signal.

    """
    return preprocess_impl(signal, fs, fmin, fmax)
