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
from typing import List, Any, Iterable

import numpy as np
from numpy import ndarray
from scipy.signal import hilbert

from pymodalib.algorithms.surrogates import surrogate_calc
from pymodalib.implementations.python.bayesian import bayes_main, CFprint, dirc
from pymodalib.implementations.python.filtering import loop_butter
from pymodalib.implementations.python.matlab_compat import sort2d
from pymodalib.utils.decorators import experimental


@experimental
def bayesian_inference(
    signal1: ndarray,
    signal2: ndarray,
    fs: float,
    interval1: Iterable[float],
    interval2: Iterable[float],
    surrogates: int,
    window: float,
    overlap: float,
    propagation_const: float,
    order: int,
    signif: float,
) -> Any:
    """
    Dynamical Bayesian inference.

    Parameters
    ----------
    signal1 : ndarray
        The first signal.
    signal2
    fs
    interval1
    interval2
    surrogates
    window
    overlap
    propagation_const
    order
    signif

    Returns
    -------
    tm
    p1
    p2
    cpl1
    cpl2
    cf1
    cf2
    mcf1
    mcf2
    surr_cpl1
    surr_cpl2
    """
    sig1 = signal1
    sig2 = signal2

    ns = surrogates

    bands1, _ = loop_butter(sig1, *interval1, fs)
    phi1 = np.angle(hilbert(bands1))

    bands2, _ = loop_butter(sig2, *interval2, fs)
    phi2 = np.angle(hilbert(bands2))

    p1 = phi1
    p2 = phi2

    #
    ##### Bayesian inference #####
    #
    tm, cc = bayes_main(
        phi1, phi2, window, 1 / fs, overlap, propagation_const, 0, order
    )

    N, s = cc.shape
    s -= 1

    cpl1 = np.zeros(N)
    cpl2 = np.zeros(N)

    q21 = np.zeros((s, s, N))
    q12 = np.zeros(q21.shape)

    for m in range(N):
        # Direction of coupling.
        cpl1[m], cpl2[m], _ = dirc(cc[m, :], order)

        # Coupling functions.
        _, _, q21[:, :, m], q12[:, :, m] = CFprint(cc[m, :], order)

    # Coupling functions for each time window.
    cf1 = q21
    cf2 = q12

    # Mean coupling functions.
    mcf1 = np.squeeze(np.mean(q21, 2))
    mcf2 = np.squeeze(np.mean(q12, 2))

    # Surrogates.
    surr1, _ = surrogate_calc(phi1, ns, "CPP", False, fs, return_params=True)
    surr2, _ = surrogate_calc(phi2, ns, "CPP", False, fs, return_params=True)

    cc_surr: List[ndarray] = []
    scpl1 = np.empty((ns, len(cc)))
    scpl2 = np.empty(scpl1.shape)

    for n in range(ns):
        _, _cc_surr = bayes_main(
            surr1[n, :],
            surr2[n, :],
            window,
            1 / fs,
            overlap,
            propagation_const,
            1,
            order,
        )
        cc_surr.append(_cc_surr)

        for idx in range(len(_cc_surr)):
            scpl1[n, idx], scpl2[n, idx], _ = dirc(_cc_surr[idx, :], order)

    alph = signif
    alph = 1 - alph / 100

    if scpl1.size > 0:
        if np.floor((ns + 1) * alph) == 0:
            surr_cpl1 = np.max(scpl1)
            surr_cpl2 = np.max(scpl2)
        else:
            K = np.floor((ns + 1) * alph)
            K = np.int(K)

            K -= 1  # Adjust for Matlab's 1-based indexing.

            s1 = sort2d(scpl1, descend=True)
            s2 = sort2d(scpl2, descend=True)

            surr_cpl1 = s1[K, :]
            surr_cpl2 = s2[K, :]
    else:
        surr_cpl1 = None
        surr_cpl2 = None

    return (
        tm,
        p1,
        p2,
        cpl1,
        cpl2,
        cf1,
        cf2,
        mcf1,
        mcf2,
        surr_cpl1,
        surr_cpl2,
    )
