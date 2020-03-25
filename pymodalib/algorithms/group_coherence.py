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
Group wavelet phase coherence with inter-subject surrogates.
"""

from typing import Tuple

from numpy import ndarray

from pymodalib.implementations.python.coherence.group_coherence import (
    dual_group_coherence as dual_group_impl,
    group_coherence as group_impl,
)


def group_coherence(
    signals_a: ndarray,
    signals_b: ndarray,
    fs: float,
    percentile: float = 95,
    max_surrogates: int = None,
    cleanup: bool = True,
    *wavelet_args,
    **wavelet_kwargs,
) -> Tuple[ndarray, ndarray, ndarray]:
    """
    Group coherence algorithm. Calculates coherences for a single group, whose members each have
    a signal A and a signal B.

    This algorithm calculates inter-subject surrogates and uses them to calculate the residual coherence.

    :param signals_a: the set of signals A for each member of the group
    :param signals_b: the signals B for each member of the group
    :param fs: the sampling frequency of the signals
    :param percentile: the percentile at which the surrogates will be subtracted from the coherence
    :param max_surrogates: the maximum number of surrogates; this may be useful to reduce the time taken
                           to perform the calculation
    :param cleanup: whether to clean up the cache folder after completion
    :return: [1D array] the frequencies;
             [2D array] the residual coherence;
             [3D array] the surrogates
    """
    return group_impl(
        signals_a,
        signals_b,
        fs=fs,
        max_surrogates=max_surrogates,
        cleanup=cleanup,
        percentile=percentile,
        *wavelet_args,
        **wavelet_kwargs,
    )


def dual_group_coherence(
    group1_signals_a: ndarray,
    group1_signals_b: ndarray,
    group2_signals_a: ndarray,
    group2_signals_b: ndarray,
    fs: float,
    percentile: float = 95,
    max_surrogates: int = None,
    *wavelet_args,
    **wavelet_kwargs,
) -> Tuple[ndarray, ndarray, ndarray, ndarray, ndarray]:
    """
    Essentially identical to 'group_coherence', except it performs group coherence on two groups instead of one.

    :return: [1D array] the frequencies;
             [2D array] the residual coherence for group 1;
             [2D array] the residual coherence for group 2;
             [3D array] the surrogates for group 1;
             [3D array] the surrogates for group 2
    """
    return dual_group_impl(
        group1_signals_a,
        group1_signals_b,
        group2_signals_a,
        group2_signals_b,
        fs=fs,
        percentile=percentile,
        max_surrogates=max_surrogates,
        *wavelet_args,
        **wavelet_kwargs,
    )
