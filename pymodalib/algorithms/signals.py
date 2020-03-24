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
import warnings

import numpy as np
from numpy import ndarray


class ResamplingException(Exception):
    pass


def resampl_flow(
    signal: ndarray, original_freq: float, resample_freq: float
) -> ndarray:
    """
    Down-samples a signal, using a moving average.

    :param signal: the signal to re-sample
    :param original_freq: the frequency of the signal
    :param resample_freq: the frequency to use for the re-sampled signal
    :return: the re-sampled signal
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
        ratio = round(ratio)

    ratio = int(ratio)
    output = np.empty(np.int(np.floor(L / ratio)))

    for j in range(0, len(output)):
        start = np.int(np.ceil(j * ratio))
        output[j] = np.mean(signal[start : start + ratio])

    return output


def moving_average(signal: ndarray, window: int, overlap: int = 0) -> ndarray:
    raise NotImplementedError("Moving average is not implemented yet.")
