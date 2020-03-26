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
Wavelet phase coherence.
"""

from typing import Tuple

import numpy as np
from numpy import ndarray


def wphcoh(wt1: ndarray, wt2: ndarray) -> Tuple[ndarray, ndarray]:
    """
    Time-averaged wavelet phase coherence.

    Parameters
    ----------
    wt1 : ndarray
        [2D array, complex] Wavelet transform of the first signal.
    wt2 : ndarray
        [2D array, complex] Wavelet transform of the second signal.

    Returns
    -------
    phcoh : ndarray
        [1D array] Wavelet phase coherence.
    phdiff : ndarray
        [1D array] Wavelet phase difference.
    """
    FN = min(wt1.shape[0], wt2.shape[0])

    wt1 = wt1[:FN, :]
    wt2 = wt2[:FN, :]

    phcoh = np.zeros((FN, 1), np.float64) * np.NaN
    phdiff = np.zeros((FN, 1), np.float64) * np.NaN

    for fn in range(FN):
        phi1 = np.angle(wt1[fn, :])
        phi2 = np.angle(wt2[fn, :])

        phexp = np.exp(1j * (phi1 - phi2))
        cphexp = phexp[~np.isnan(phexp)]

        wt1_i = wt1[fn]
        wt2_i = wt2[fn]

        cutoff = min(wt1.shape[1], wt2.shape[1])

        indices_where_zero = ((wt1_i[:cutoff] == 0) & (wt2_i[:cutoff] == 0)).nonzero()
        NL = len(indices_where_zero[0])

        CL = cphexp.shape[0]
        if CL > 0:
            phph = np.mean(cphexp) - NL / CL
            phcoh[fn] = np.abs(phph)
            phdiff[fn] = np.angle(phph)

    return phcoh, phdiff


def tlphcoh(
    wt1: ndarray, wt2: ndarray, freq: ndarray, fs: float, wsize: int = 10
) -> ndarray:
    """
    Time-localized phase coherence.

    Parameters
    ----------
    wt1 : ndarray
        [2D array, complex] Wavelet transform of the first signal.
    wt2 : ndarray
        [2D array, complex] Wavelet transform of the second signal. Must be the same shape as `wt1`.
    freq : ndarray
        [1D array] Frequencies at which the wavelet transforms wt1 and wt2 were calculated.
    fs : float
        Sampling frequency of the signals.
    wsize : int, optional
         (Default value = 10) The window size.

    Returns
    -------
    tpc : ndarray
        [2D array] The time-localized wavelet phase coherence.
    """
    NF, L = wt1.shape

    ipc = np.exp(1j * np.angle(wt1 * wt2.conj()))
    zpc = ipc.copy()
    zpc[np.isnan(zpc)] = 0

    zeros = np.zeros((NF, 1), np.complex64)
    csum = np.cumsum(zpc, axis=1)

    cum_pc = np.hstack((zeros, csum))
    tpc = np.zeros((NF, L), np.complex64) * np.NaN

    for fn in range(NF):
        cs = ipc[fn, :]
        cumcs = cum_pc[fn, :]

        f = np.nonzero(~np.isnan(cs))[0]
        window = np.round(wsize / freq[fn] * fs)
        window += 1 - np.mod(window, 2)

        hw = np.floor(window / 2)

        if len(f) >= 2:
            tn1 = f[0]
            tn2 = f[-1]
            if window <= tn2 - tn1:
                window = np.int(window)
                hw = np.int(hw)

                locpc = (
                    np.abs(
                        cumcs[tn1 + window : tn2 + 1 + 1]
                        - cumcs[tn1 : tn2 - window + 1 + 1]
                    )
                    / window
                )
                tpc[fn, tn1 + hw : tn2 - hw + 1] = locpc

    return tpc  # TODO: implement 'under_sample' from matlab version?
