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
import multiprocessing
import warnings
from typing import Any

import numpy as np
from numpy import ndarray

from pymodalib.algorithms.wavelet import wavelet_transform


class CoherenceException(Exception):
    pass


def avg_wt(signal: ndarray, fs: float, *args, **kwargs):
    wt, freq = wavelet_transform(signal, fs, *args, **kwargs)

    ampl = np.abs(wt)
    return np.average(ampl, axis=1)


def _group_avg_wt(signals: ndarray, fs: float, *args, **kwargs):
    out = None

    x, y = signals.shape
    for index in range(x):
        avg = avg_wt(signals[index, :], fs, *args, **kwargs)

        if out is None:
            out = np.empty((x, len(avg)))

        out[index, :] = avg[:]

    return out


def group_coherence(
    signals_a: ndarray,
    signals_b: ndarray,
    fs: float,
    max_surrogates: int = None,
    *wavelet_args,
    **wavelet_kwargs,
) -> ndarray:
    """
    Group coherence algorithm.

    :param signals_a:
    :param signals_b:
    :param fs:
    :param max_surrogates:
    :return: # TODO
    """
    try:
        xa, ya = signals_a.shape
        xb, yb = signals_a.shape
    except ValueError:
        raise CoherenceException(
            f"Cannot perform group coherence with only one pair of signals."
        )

    if xa > ya:
        warnings.warn(
            f"Array dimensions {xa}, {ya} imply that the signals may be orientated incorrectly in the input arrays. "
            f"If this is not the case, please ignore the warning.",
            RuntimeWarning,
        )

    processes = multiprocessing.cpu_count() + 1
    pool = multiprocessing.Pool(processes=processes)

    args = [(signals_a[0, :], fs), (signals_b[0, :], fs)]
    wt_a, wt_b = pool.starmap(avg_wt, args)

    wavelet_transforms_a = np.empty((xa, len(wt_a)))
    wavelet_transforms_b = np.empty((xb, len(wt_b)))

    indices = np.arange(1, xa)
    chunks = np.array_split(indices, processes)

    args = [(signals_a[chunk[0] : chunk[-1], :], fs,) for chunk in chunks]
    results = pool.starmap(_group_avg_wt, args)

    for chunk, result in zip(chunks, results):
        start, end = chunk[0], chunk[-1]

        wavelet_transforms_a[start:end, :] = result[:, :]
        wavelet_transforms_b[start:end, :] = result[:, :]


def dual_group_coherence(
    group1_signals1: ndarray,
    group1_signals2: ndarray,
    group2_signals1: ndarray,
    group2_signals2: ndarray,
    percentile: float = 95,
    max_surrogates: int = None,
    *wavelet_args,
    **wavelet_kwargs,
) -> Any:
    """
    Group coherence algorithm. Uses inter-subject surrogates.

    :param group1_signals1:
    :param group1_signals2:
    :param group2_signals1:
    :param group2_signals2:
    :param percentile:
    :param max_surrogates:
    :return:
    """
    try:
        x1a, y1a = group1_signals1.shape
        x1b, y1b = group1_signals2.shape
        x2a, y2a = group2_signals1.shape
        x2b, y2b = group2_signals2.shape

    except ValueError:
        raise CoherenceException(
            f"Cannot perform group coherence if multiple signals are not present in each group."
        )

    if x1a > y1a:
        warnings.warn(
            f"Array dimensions {x1a}, {y1a} imply that the signals may be orientated incorrectly in the input arrays. "
            f"If this is not the case, please ignore the warning.",
            RuntimeWarning,
        )

    if x1a != x1b or y1a != y1b or x2a != x2b or y2a != y2b:
        raise CoherenceException(
            f"Dimensions of input arrays do not match. "
            f"The dimensions of each group's signals A and signals B must be the same.",
            RuntimeWarning,
        )

    recommended_surr = 19
    surr = y1b

    if max_surrogates < recommended_surr:
        warnings.warn(
            f"Low number of surrogates: {max_surrogates}. A larger number of surrogates is recommended.",
            RuntimeWarning,
        )

    """
    Create array containing wavelet transforms.
    """
    group1_wt1 = None
    group1_wt2 = None
    group2_wt1 = None
    group2_wt2 = None

    """
    Create array containing 
    """
