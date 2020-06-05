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
from typing import List, Any, Iterable, Tuple

import numpy as np
from numpy import ndarray
from scipy.signal import hilbert

from pymodalib.algorithms.surrogates import surrogate_calc
from pymodalib.implementations.python.filtering import loop_butter
from pymodalib.implementations.python.matlab_compat import (
    twopi,
    backslash,
    is_arraylike,
    sort2d,
)
from pymodalib.utils.decorators import experimental

"""
Translation of the MODA Bayesian inference algorithm into Python.

STATUS: Finished, but not working. Current issue: `filtfilt` in `loop_butter` is
not giving accurate results.

UPDATE: The problem may be in the value of `cc` being incorrect.
"""


@experimental
def bayesian_inference_impl(
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
    *args,
    **kwargs
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


def bayes_main(ph1, ph2, win, h, ovr, pr, s, bn) -> Tuple[ndarray, ndarray]:
    win /= h

    w = ovr * win
    ps = ph2 - ph1
    pw = win * h * pr

    M = 2 + 2 * ((2 * bn + 1) ** 2 - 1)
    L = 2

    M = np.int(M)
    Cpr = np.zeros((int(M / L), L))
    XIpr = np.zeros((M, M))

    if max(ph1) < twopi + 0.1:
        ph1 = np.unwrap(ph1)
        ph2 = np.unwrap(ph2)

    if len(ph1.shape) > 1:
        m, n = ph1.shape
    else:
        m = ph1.shape[0]
        n = 1

    if m < n:
        ph1 = ph1.conj().T
        ph2 = ph2.conj().T

    s = np.int(np.ceil((len(ps) - win) / w))
    # e = zeros((s, s, s))

    w = int(w)
    win = int(win)

    ph1 = ph1.reshape(ph1.shape[0])
    ph2 = ph2.reshape(ph2.shape[0])

    if len(ps.shape) == 2:
        ps = ps.reshape(ps.shape[1])

    r = int(np.floor((len(ps) - win) / w)) + 1
    cc = np.zeros((r, Cpr.size))

    for i in range(r):
        phi1 = ph1[i * w : i * w + win]
        phi2 = ph2[i * w : i * w + win]

        # Note: certain values of Cpt are incorrect, while some are accurate.
        Cpt, XIpt, E = bayesPhs(Cpr, XIpr, h, 500, 1e-5, phi1, phi2, bn)

        XIpr, Cpr = propagation_function_XIpt(Cpt, XIpt, pw)

        # e[i, :, :] = E
        cc[i, :] = np.concatenate([Cpt[:, i] for i in range(Cpt.shape[1])])

    tm = np.arange(win / 2, len(ph1) - win / 2, w) * h

    # Note: 'e' was removed because it was causing a bug and is never used.
    return tm, cc  # , e


def propagation_function_XIpt(Cpt, XIpt, p) -> Tuple[ndarray, ndarray]:
    Cpr = Cpt

    Inv_Diffusion = np.zeros((len(XIpt), len(XIpt)))
    invXIpt = np.linalg.inv(XIpt)

    for i in range(Cpt.size):
        Inv_Diffusion[i, i] = p ** 2 * invXIpt[i, i]

    XIpr = np.linalg.inv(invXIpt + Inv_Diffusion)

    return XIpr, Cpr


def bayesPhs(
    Cpr, XIpr, h, max_loops, eps, phi1, phi2, bn
) -> Tuple[ndarray, ndarray, ndarray]:
    phi1S = (phi1[1:] + phi1[:-1]) / 2
    phi2S = (phi2[1:] + phi2[:-1]) / 2

    phi1T = (phi1[1:] - phi1[:-1]) / h
    phi2T = (phi2[1:] - phi2[:-1]) / h

    phiT = np.asarray([phi1T, phi2T])

    L = 2
    M = 2 + 2 * ((2 * bn + 1) ** 2 - 1)
    K = M / L

    p = calculateP(phi1S, phi2S, K, bn)
    v1 = calculateV(phi1S, phi2S, K, bn, 1)
    v2 = calculateV(phi1S, phi2S, K, bn, 2)

    C_old = Cpr

    Cpt = Cpr.copy()
    XIpt = None
    E = None

    for loop in range(max_loops):
        E = calculateE(Cpt.conj().T, phiT, L, h, p)
        Cpt, XIpt = calculateC(E, p, v1, v2, Cpr, XIpr, M, L, phiT, h)

        if np.sum((C_old - Cpt) * (C_old - Cpt) / (Cpt ** 2)) < eps:
            return Cpt, XIpt, E

        C_old = Cpt

    return Cpt, XIpt, E


def calculateP(phi1, phi2, K, bn) -> ndarray:
    bn = np.int(bn)
    K = np.int(K)

    p = np.zeros((K, len(phi1)))

    p[0, :] = 1
    br = 1

    for i in range(1, bn + 1):
        p[br, :] = np.sin(i * phi1)
        p[br + 1, :] = np.cos(i * phi1)

        br += 2

    for i in range(1, bn + 1):
        p[br, :] = np.sin(i * phi2)
        p[br + 1, :] = np.cos(i * phi2)
        br += 2

    for i in range(1, bn + 1):
        for j in range(1, bn + 1):
            p[br, :] = np.sin(i * phi1 + j * phi2)
            p[br + 1, :] = np.cos(i * phi1 + j * phi2)
            br += 2

            p[br, :] = np.sin(i * phi1 - j * phi2)
            p[br + 1, :] = np.cos(i * phi1 - j * phi2)
            br += 2

    return p


def calculateV(phi1, phi2, K, bn, mr) -> ndarray:
    bn = np.int(bn)
    K = np.int(K)
    v = np.zeros((K, len(phi1)))

    br = 1

    if mr == 1:
        for i in range(1, bn + 1):
            v[br, :] = i * np.cos(i * phi1)
            v[br + 1, :] = -i * np.sin(i * phi1)

            br += 2

        for i in range(1, bn + 1):
            v[br, :] = 0
            v[br + 1, :] = 0
            br += 2

        for i in range(1, bn + 1):
            for j in range(1, bn + 1):
                v[br, :] = i * np.cos(i * phi1 + j * phi2)
                v[br + 1, :] = -i * np.sin(i * phi1 + j * phi2)
                br += 2

                v[br, :] = i * np.cos(i * phi1 - j * phi2)
                v[br + 1, :] = -i * np.sin(i * phi1 - j * phi2)
                br += 2
    else:
        for i in range(1, bn + 1):
            v[br, :] = 0
            v[br + 1, :] = 0

            br += 2

        for i in range(1, bn + 1):
            v[br, :] = i * np.cos(i * phi2)
            v[br + 1, :] = -i * np.sin(i * phi2)

            br += 2

        for i in range(1, bn + 1):
            for j in range(1, bn + 1):
                v[br, :] = j * np.cos(i * phi1 + j * phi2)
                v[br + 1, :] = -j * np.sin(i * phi1 + j * phi2)
                br += 2

                v[br, :] = -j * np.cos(i * phi1 - j * phi2)
                v[br + 1, :] = j * np.sin(i * phi1 - j * phi2)
                br += 2

    return v


def calculateE(c, phiT, L, h, p) -> ndarray:
    E = np.zeros((L, L))

    mul = c @ p
    sub = phiT - mul
    E += sub @ sub.conj().T
    E = (h / len(phiT[0, :])) * E

    return E


def calculateC(E, p, v1, v2, Cpr, XIpr, M, L, phiT, h) -> Tuple[ndarray, ndarray]:
    K = M / L
    invr = np.linalg.inv(E)

    K = int(K)
    M = int(M)

    XIpt = np.zeros((M, M))
    Cpt = np.zeros(Cpr.shape)

    mul = p @ p.conj().T

    XIpt[:K, :K] = XIpr[:K, :K] + h * invr[0, 0] * mul
    XIpt[:K, K : 2 * K] = XIpr[:K, K : 2 * K] + h * invr[0, 1] * mul
    XIpt[K : 2 * K, :K] = XIpr[K : 2 * K, :K] + h * invr[1, 0] * mul
    XIpt[K : 2 * K, K : 2 * K] = XIpr[K : 2 * K, K : 2 * K] + h * invr[1, 1] * mul

    # Evaluate from temp r.
    r = np.zeros((K, L))
    ED = backslash(E, phiT)

    sum_v1 = np.sum(v1, axis=1)
    sum_v2 = np.sum(v2, axis=1)
    # sum_v1 = sum_v1.reshape(len(sum_v1), 1)

    r[:, 0] = (
        XIpr[:K, :K] @ Cpr[:, 0]
        + XIpr[:K, K : 2 * K] @ Cpr[:, 1]
        + h * ((p @ ED[0, :].conj().T) - 0.5 * sum_v1)
    )

    r[:, 1] = (
        XIpr[K : 2 * K, :K] @ Cpr[:, 0]
        + (XIpr[K : 2 * K, K : 2 * K] @ Cpr[:, 1])
        + h * ((p @ ED[1, :].conj().T) - 0.5 * sum_v2)
    )

    C = backslash(XIpt, np.concatenate([r[:, 0], r[:, 1]])).conj().T
    Cpt[:, 0] = C[:K]
    Cpt[:, 1] = C[K : 2 * K]

    return Cpt, XIpt


def dirc(c, bn) -> Tuple[ndarray, ndarray, ndarray]:
    q1 = np.zeros((bn * 8,))
    q2 = np.zeros(q1.shape)
    iq1 = 0
    iq2 = 0
    br = 1
    K = np.int(len(c) / 2)

    if is_arraylike(c[0]):
        c = c[0]

    for ii in range(bn):
        q1[iq1] = c[br]
        iq1 = iq1 + 1
        q1[iq1] = c[br + 1]
        iq1 = iq1 + 1
        q2[iq2] = c[K + br]
        iq2 = iq2 + 1
        q2[iq2] = c[K + br + 1]
        iq2 = iq2 + 1
        br = br + 2

    for ii in range(bn):
        q1[iq1] = c[br]
        iq1 = iq1 + 1
        q1[iq1] = c[br + 1]
        iq1 = iq1 + 1
        q2[iq2] = c[K + br]
        iq2 = iq2 + 1
        q2[iq2] = c[K + br + 1]
        iq2 = iq2 + 1
        br = br + 2

    for jj in range(bn):
        for ii in range(bn):
            q1[iq1] = c[br]
            iq1 = iq1 + 1
            q1[iq1] = c[br + 1]
            iq1 = iq1 + 1
            q2[iq2] = c[K + br]
            iq2 = iq2 + 1
            q2[iq2] = c[K + br + 1]
            iq2 = iq2 + 1
            br = br + 2

    cpl1 = np.linalg.norm(q1)
    cpl2 = np.linalg.norm(q2)
    drc = (cpl2 - cpl1) / (cpl1 + cpl2)

    return cpl1, cpl2, drc


def CFprint(cc, bn) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    t1 = np.arange(0, twopi, 0.13)
    t2 = t1.copy()

    q1 = np.zeros((len(t1), len(t1)))
    q1[: len(t1), : len(t1)] = 0

    q2 = q1.copy()

    u = cc
    K = np.int(len(u) / 2)

    for i1 in range(len(t1)):
        for j1 in range(len(t2)):
            br = 1

            for ii in range(1, bn + 1):
                q1[i1, j1] = (
                    q1[i1, j1]
                    + u[br] * np.sin(ii * t1[i1])
                    + u[br + 1] * np.cos(ii * t1[i1])
                )
                q2[i1, j1] = (
                    q2[i1, j1]
                    + u[K + br] * np.sin(ii * t2[j1])
                    + u[K + br + 1] * np.cos(ii * t2[j1])
                )

                br += 2

            for ii in range(1, bn + 1):
                q1[i1, j1] = (
                    q1[i1, j1]
                    + u[br] * np.sin(ii * t2[j1])
                    + u[br + 1] * np.cos(ii * t2[j1])
                )
                q2[i1, j1] = (
                    q2[i1, j1]
                    + u[K + br] * np.sin(ii * t1[i1])
                    + u[K + br + 1] * np.cos(ii * t1[i1])
                )

                br += 2

            for ii in range(1, bn + 1):
                for jj in range(1, bn + 1):
                    q1[i1, j1] = (
                        q1[i1, j1]
                        + u[br] * np.sin(ii * t1[i1] + jj * t2[j1])
                        + u[br + 1] * np.cos(ii * t1[i1] + jj * t2[j1])
                    )

                    q2[i1, j1] = (
                        q2[i1, j1]
                        + u[K + br] * np.sin(ii * t1[i1] + jj * t2[j1])
                        + u[K + br + 1] * np.cos(ii * t1[i1] + jj * t2[j1])
                    )

                    br += 2

                    q1[i1, j1] = (
                        q1[i1, j1]
                        + u[br] * np.sin(ii * t1[i1] - jj * t2[j1])
                        + u[br + 1] * np.cos(ii * t1[i1] - jj * t2[j1])
                    )

                    q2[i1, j1] = (
                        q2[i1, j1]
                        + u[K + br] * np.sin(ii * t1[i1] - jj * t2[j1])
                        + u[K + br + 1] * np.cos(ii * t1[i1] - jj * t2[j1])
                    )

                    br += 2

    return t1, t2, q1, q2
