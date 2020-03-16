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

from algorithms.coherence import wavelet_phase_coherence
from pymodalib.algorithms.wavelet import wavelet_transform


class CoherenceException(Exception):
    pass


def wt(signal: ndarray, fs: float, *args, **kwargs):
    wt, freq = wavelet_transform(signal, fs, *args, **kwargs)
    return wt, freq


def _group_wt(signals: ndarray, fs: float, *args, **kwargs):
    out = None

    x, y = signals.shape
    for index in range(x):
        _wt, _ = wt(signals[index, :], fs, *args, **kwargs)

        if out is None:
            out = np.empty((x, *_wt.shape), dtype=np.complex64)

        out[index, :, :] = _wt[:, :]

    return out


def _group_coherence(wavelet_transforms_a: ndarray, wavelet_transforms_b: ndarray):
    coh_length = wavelet_transforms_a.shape[1]
    out = np.empty((len(wavelet_transforms_a), len(wavelet_transforms_b), coh_length))

    for i, wt1 in enumerate(wavelet_transforms_a):
        for j, wt2 in enumerate(wavelet_transforms_b):
            coh, _ = wavelet_phase_coherence(wt1, wt2)
            out[i, j, :] = np.average(coh, axis=1)

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

    # Calculate the first two wavelet transforms, so that we know their dimensions.
    args = [(signals_a[0, :], fs), (signals_b[0, :], fs)]
    (wt_a, _), (wt_b, _) = pool.starmap(wt, args)

    # Create empty arrays for all wavelet transforms.
    wavelet_transforms_a = np.empty((xa, *wt_a.shape), dtype=np.complex64)
    wavelet_transforms_b = np.empty((xb, *wt_b.shape), dtype=np.complex64)

    # Calculate how the signals will be split up,
    # so each process can work on part of the group.
    indices = np.arange(1, xa)
    chunks = np.array_split(indices, processes)

    # Calculate wavelet transforms in parallel.
    args = [(signals_a[chunk[0] : chunk[-1], :], fs,) for chunk in chunks]
    results = pool.starmap(_group_wt, args)

    # Write the results from processes into the arrays containing the wavelet transforms.
    for chunk, result in zip(chunks, results):
        start, end = chunk[0], chunk[-1]

        wavelet_transforms_a[start:end, :, :] = result[:, :, :]
        wavelet_transforms_b[start:end, :, :] = result[:, :, :]

    """
    Now we have the wavelet transform for every signal in the group.
    
    Next, we want to calculate the coherence between every signal A and B. 
    
    The group will have an array like the following, where the empty items 
    contain the coherence between their associated signal A and B:
    
                |  sig_a_1  |  sig_a_2  |  sig_a_3  | ..... |
    | --------- | --------- | --------- | --------- | ----- |
    |  sig_b_1  |           |           |           |       |
    |  sig_b_2  |           |           |           |       |
    |  sig_b_3  |           |           |           |       |
    |  .......  |           |           |           |       |
    """

    # Create empty array for coherence.
    coherence = np.empty((xa, *wavelet_transforms_a.shape[:-1]))

    indices = np.arange(0, coherence.shape[0])
    chunks = np.array_split(indices, processes)

    args = []
    for c in chunks:
        start, end = c[0], c[-1]

        # Only split up rows. Keep columns the same.
        wavelets_a = wavelet_transforms_a[start:end, :, :]
        wavelets_b = wavelet_transforms_b[:, :, :]

        args.append((wavelets_a, wavelets_b))

    results = pool.starmap(_group_coherence, args)

    for chunk, result in zip(chunks, results):
        start, end = chunk[0], chunk[-1]

        coherence[start:end, :] = result[:, :]

    """
    Now we have a large array containing the coherence between all signals.
    
    The values on the diagonal are the useful coherences; the other values are
    surrogates.
    """


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
