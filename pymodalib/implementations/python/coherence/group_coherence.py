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
import uuid
import warnings
from typing import Any, Tuple

import numpy as np
import pymodalib
from numpy import ndarray

from pymodalib.algorithms.coherence import wavelet_phase_coherence
from pymodalib.algorithms.wavelet import wavelet_transform
from pymodalib.utils.chunks import array_split


class CoherenceException(Exception):
    pass


def wt(signal: ndarray, fs: float, *args, **kwargs):
    wt, freq = wavelet_transform(signal, fs, *args, **kwargs, Display="off")
    return wt, freq


def _chunk_wt(signals_a: ndarray, signals_b: ndarray, fs: float, *args, **kwargs):
    out_a = None
    out_b = None

    x, y = signals_a.shape
    for index in range(x):
        _wt_a, _ = wt(signals_a[index, :], fs, *args, **kwargs)
        _wt_b, _ = wt(signals_b[index, :], fs, *args, **kwargs)

        if out_a is None:
            out_a = pymodalib.cachedarray(shape=(x, *_wt_a.shape), dtype=np.complex64)
            out_b = pymodalib.cachedarray(shape=(x, *_wt_a.shape), dtype=np.complex64)

        out_a[index, :, :] = _wt_a[:, :]
        out_b[index, :, :] = _wt_b[:, :]

    return out_a, out_b


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
) -> Tuple[ndarray, ndarray, ndarray]:
    """
    Group coherence algorithm. Calculates coherences for a single group, which contains a
    signal A and a signal B for each member.

    Performs inter-subject surrogates.

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

    # Create Pool for multiprocessing.
    processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=processes)

    # Calculate the first two wavelet transforms, so we know their dimensions.
    args = [(signals_a[0, :], fs), (signals_b[0, :], fs)]

    (wt_a, freq), (wt_b, _) = pool.starmap(wt, args)
    print(f"Finished calculating first pair of wavelet transforms.")

    # Create empty arrays to hold all wavelet transforms.
    wavelet_transforms_a = pymodalib.cachedarray(
        shape=(xa, *wt_a.shape), dtype=np.complex64
    )
    wavelet_transforms_b = pymodalib.cachedarray(
        shape=(xb, *wt_b.shape), dtype=np.complex64
    )

    # Calculate how the signals will be split up, so each process can work on part of the group.
    indices = np.arange(1, xa)
    chunks = array_split(indices, processes)

    args = []
    for c in chunks:
        start, end = c[0], c[-1]
        if start == end:
            end += 1

        args.append((signals_a[start:end, :], signals_b[start:end, :], fs))

    # Calculate wavelet transforms in parallel.
    results = pool.starmap(_chunk_wt, args)
    print(f"Finished calculating wavelet transforms.")

    # Write the results from processes into the arrays containing the wavelet transforms.
    for chunk, (result1, result2) in zip(chunks, results):
        start, end = chunk[0], chunk[-1] + 1

        wavelet_transforms_a[start:end, :, :] = result1[:, :, :]
        wavelet_transforms_b[start:end, :, :] = result2[:, :, :]

    """
    Now we have the wavelet transform for every signal in the group.
    
    Next, we want to calculate the coherence between every signal A and B. 
    The coherences between unrelated signals (e.g. signal A1 and signal B2)
    will be used as surrogates.
    
    The group will have a coherence array like the following, where the cells marked 
    with "C" are the coherences between the signals for each subject in the group,
    and the cells marked with "s" are the surrogates created by calculating the 
    coherence between unrelated signals.
    
                |  sig_b_1  |  sig_b_2  |  sig_b_3  |  .....  |
    | --------- | --------- | --------- | --------- |  -----  |
    |  sig_a_1  |     C     |     s     |     s     |    s    |
    |  sig_a_2  |     s     |     C     |     s     |    s    |
    |  sig_a_3  |     s     |     s     |     C     |    s    |
    |   .....   |     s     |     s     |     s     |    C    |
    """

    # Create empty array for coherence.
    coherence = np.empty((xa, *wavelet_transforms_a.shape[:-1]))

    indices = np.arange(0, coherence.shape[0])
    chunks = array_split(indices, processes)

    args = []
    for c in chunks:
        start, end = c[0], c[-1] + 1

        # Split up into chunks by rows.
        wavelets_a = wavelet_transforms_a[start:end, :, :]

        # Keep all columns instead of splitting into chunks.
        wavelets_b = wavelet_transforms_b[:, :, :]

        args.append((wavelets_a, wavelets_b))

    del wavelet_transforms_a
    del wavelet_transforms_b

    results = pool.starmap(_group_coherence, args)
    print(f"Finished calculating coherence.")

    # Write the results from processes into the coherence array.
    for chunk, result in zip(chunks, results):
        start, end = chunk[0], chunk[-1] + 1

        coherence[start:end, :] = result[:, :]

    """
    Now we have a large array containing the coherence between all signals.
    
    The values on the diagonal are the useful coherences; the other values are
    surrogates.
    """

    # Indices along the diagonal of the coherence array.
    # These correspond to the useful coherences.
    diag = np.diag_indices(xa)
    real_coherences = coherence[diag]

    # Set the coherences to NaN, so we're left with the surrogates only.
    coherence[diag] = np.NaN
    surrogates = coherence

    surr_percentile = np.nanpercentile(surrogates, 95, axis=(0, 1,))

    residual_coherence = real_coherences - surr_percentile
    residual_coherence[residual_coherence < 0] = 0

    return freq, residual_coherence, surr_percentile


def dual_group_coherence(
    group1_signals1: ndarray,
    group1_signals2: ndarray,
    group2_signals1: ndarray,
    group2_signals2: ndarray,
    fs: float,
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
    :param fs:
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

    if max_surrogates and max_surrogates < recommended_surr:
        warnings.warn(
            f"Low number of surrogates: {max_surrogates}. A larger number of surrogates is recommended.",
            RuntimeWarning,
        )

    result1 = group_coherence(
        group1_signals1, group1_signals2, fs, *wavelet_args, **wavelet_kwargs
    )
    result2 = group_coherence(
        group1_signals1, group1_signals2, fs, *wavelet_args, **wavelet_kwargs
    )

    return (*result1, *result2)
