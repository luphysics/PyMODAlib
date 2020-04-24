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

from typing import Tuple, List

import numpy as np
from numpy import ndarray
from scipy.stats import ranksums

from pymodalib.implementations.python.coherence.group_coherence import (
    dual_group_coherence_impl as dual_group_impl,
    group_coherence_impl as group_impl,
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
    Group wavelet phase coherence algorithm. Calculates coherence for a single group, whose members each
    have a signal A and a signal B.

    This algorithm uses inter-subject surrogates to calculate the residual coherence.

    .. note::
        You can also pass *args and **kwargs to this function, which will be used when
        performing the wavelet transform. For example, `wavelet=”Morlet”`.

    Parameters
    ----------
    signals_a : ndarray
        [2D array] The set of signals A for each member of the group.
    signals_b : ndarray
        [2D array] The set of signals B for each member of the group.
    fs : float
        The sampling frequency of the signals.
    percentile : float, optional
        (Default value = 95) The percentile of the surrogates which will be subtracted from the coherence.
    max_surrogates : int, optional
        (Default value = None) The maximum number of surrogates to use. You should usually leave this unchanged, but you may wish to use it
        to reduce the time taken to perform the calculation.
    cleanup : bool, optional
        (Default value = True) Whether to clean up the cache folder after completion.
    wavelet_args : Any, optional
        Arguments to pass to the wavelet transform function.
    wavelet_kwargs : Any, optional
        Keyword arguments to pass to the wavelet transform function.

    Returns
    -------
    freq : ndarray
        [1D array] The frequencies.
    coh : ndarray
        [2D array] The residual coherence.
    surr : ndarray
        [3D array] The surrogates.
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
    Group wavelet phase coherence algorithm. Calculates coherence for two groups, whose members each have
    a signal A and a signal B. The groups can be different sizes.

    This algorithm calculates inter-subject surrogates and uses them to calculate the residual coherence.

    .. note::
        you can also pass *args and **kwargs to this function, which will be used when
        performing the wavelet transform. For example, `wavelet="Morlet"`.

    Parameters
    ----------
    group1_signals_a : ndarray
        [2D array] The set of signals A for group 1.
    group1_signals_b : ndarray
        [2D array] The set of signals B for group 1.
    group2_signals_a : ndarray
        [2D array] The set of signals A for group 2.
    group2_signals_b : ndarray
        [2D array] The set of signals B for group 2.
    fs : float
        The sampling frequency of the signals.
    percentile : float, optional
        (Default value = 95) The percentile of the surrogates which will be subtracted from the coherence.
    max_surrogates : int, optional
        (Default value = None) The maximum number of surrogates to use. You should usually leave this unchanged,
        but you may wish to use it to reduce the time taken to perform the calculation.
    wavelet_args : Any, optional
        Arguments to pass to the wavelet transform function.
    wavelet_kwargs : Any, optional
        Keyword arguments to pass to the wavelet transform function.

    Returns
    -------
    freq : ndarray
        [1D array] The frequencies.
    coh1 : ndarray
        [2D array] The residual coherence for group 1.
    coh2 : ndarray
        [2D array] The residual coherence for group 2.
    surr1 : ndarray
        [3D array] The surrogates for group 1.
    surr2 : ndarray
        [3D array] The surrogates for group 2.
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


def statistical_test(
    freq: ndarray, coh1: ndarray, coh2: ndarray, bands: List[Tuple[float, float]]
) -> List[float]:
    """
    Performs a statistical test on the results of dual group wavelet phase coherence, to check for significance.

    Parameters
    ----------
    freq : ndarray
        [1D array] The frequencies.
    coh1 : ndarray
        [2D array] The coherence for group 1.
    coh2 : ndarray
        [2D array] The coherence for group 2.
    bands : List[Tuple[int, int]]
        The frequency intervals which the statistics will be calculated for.

    Returns
    -------
    pvalues : List[float]
        The p-values associated with each frequency interval.
    """
    assert bands, f"At least one frequency band must be supplied."
    assert len(freq) == coh1.shape[1] == coh2.shape[1], (
        f"The first dimension of 'freq' must be equal to the "
        f"second dimension of 'coh1' and 'coh2'."
    )

    indices = []
    for b in bands:
        f1, f2 = b
        band_indices = ((freq >= f1) & (freq < f2)).nonzero()[0]
        indices.append(band_indices)

    results: List[Tuple[float, float]] = []

    for index_arr in indices:
        x = coh1[:, index_arr]
        y = coh2[:, index_arr]

        xm = np.nanmean(x, axis=1)
        ym = np.nanmean(y, axis=1)

        if xm.size > 0 and not np.isnan(xm).all() and not np.isnan(ym).all():
            w, p = ranksums(xm, ym)
        else:
            w, p = np.NaN, np.NaN

        results.append((w, p,))

    return [p for (w, p) in results]
