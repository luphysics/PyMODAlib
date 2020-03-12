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
from typing import Optional

from numpy import ndarray

from pymodalib.implementations.python.signals.preprocessing import preprocess_impl


def preprocess(
    signal: ndarray,
    fs: float,
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
) -> ndarray:
    """
    Pre-processes a signal, performing de-trending.

    :param signal: the signal to pre-process
    :param fs: the sampling frequency
    :param fmin: the minimum frequency
    :param fmax: the maximum frequency
    :return: the pre-processed signal as a new ndarray
    """
    return preprocess_impl(signal, fs, fmin, fmax)
