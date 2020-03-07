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
from typing import Tuple

from numpy import ndarray

from pymodalib.implementations.python.harmonics.harmonics import (
    harmonicfinder_impl_python as impl,
)


def harmonicfinder(
    signal: ndarray,
    fs: float,
    scale_min: float,
    scale_max: float,
    sigma: float = 1.05,
    time_resolution: float = 0.1,
    surrogates: int = 10,
    parallel: bool = True,
) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    """
    Harmonic-finder algorithm.

    :param signal: [1D array] the signal to analyse
    :param fs: the sampling frequency of the signal
    :param scale_min:                                   #TODO docs
    :param scale_max:                                   #TODO docs
    :param sigma:                                       #TODO docs
    :param time_resolution: the time resolution
    :param surrogates: the number of surrogates
    :param parallel: whether to parallelize the algorithm, which provides a significant speed boost in many cases
    :return: [1D array] the frequencies;                #TODO docs
             [2D array] the raw harmonics;
             [2D array] the number of surrogates which the raw harmonics are higher than at each point;
             [2D array] the raw harmonics relative to the mean and standard deviation of the surrogate distribution
    """
    return impl(
        signal, fs, scale_min, scale_max, sigma, time_resolution, surrogates, parallel
    )
