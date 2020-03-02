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
Harmonics finder, translated from the MATLAB code which was written by Lawrence Sheppard.
"""

from typing import Tuple

import numpy as np
from numpy import ndarray


def harmonicfinder(
    signal: ndarray,
    fs: float,
    scale_min: float,
    scale_max: float,
    sigma: float,
    time_res: float,
    surr_count: int,
) -> Tuple[ndarray, ndarray]:
    trans1 = modbasicwavelet_flow_cmplx4(
        signal, fs, scale_min, scale_max, sigma, time_res
    )

    scalefrequency1 = scale_frequency(scale_min, scale_max, sigma)

    m, n = trans1.shape
    detsig = signal

    res = np.empty((m, m,)).fill(np.NaN)

    for a1 in range(m):
        margin = np.ceil((np.sum(np.isnan(np.angle(trans1[a1, :n])))) / 2)
        phase1 = np.angle(trans1[a1, margin : n - margin])  # Slow.

        for a2 in range(a1):
            phase2 = np.angle(trans1[a2, margin : n - margin])  # Fast.
            if a2 == a1 and np.any(phase1 - phase2) == 1:
                error = True

            binner, mean_index = indexfinder3(phase1, phase2)
            res[a1, a2] = mean_index
            res[a2, a1] = mean_index

    ressur = np.empty((surr_count, m, m,)).fill(np.NaN)
    for sigb in range(surr_count):
        surrsig = aaft4(detsig.conj().T)

        transsurr = modbasicwavelet_flow_cmplx4(
            surrsig, scale_min, scale_max, sigma, time_res
        )

        for a1 in range(m):
            margin = np.ceil((np.sum(np.isnan(np.angle(trans1[a1, :n])))) / 2)
            phase1 = np.angle(trans1[a1, margin : n - margin])  # Slow.

            for a2 in range(a1):
                phase2 = np.angle(trans1[a2, margin : n - margin])  # Fast.
                if a2 == a1 and np.any(phase1 - phase2) == 1:
                    error = True

                binner, mean_index = indexfinder3(phase1, phase2)
                res[a1, a2] = mean_index
                res[a2, a1] = mean_index

    for a1 in range(m):
        for a2 in range(a1):
            ysurr, isurr = np.argsort(np.asarray((res[a1, a2], ressur[:, a1, a2],)))
            sig[isurr] = np.arange(0, surr_count + 1)

            pos[a1, a2] = sig[0]
            pos[a2, a1] = sig[0]

    for a1 in range(m):
        for a2 in range(m):
            surrmean[a1, a2] = np.nanmean(ressur[:, a1, a2])
            surrstd[a1, a2] = np.nanstd(ressur[:, a1, a2])
            sig = (res[a1, a2] - surrmean[a1, a2]) / surrstd[a1, a2]
            pos[a1, a2] = np.min(sig, 5)


def scale_frequency(scale_min: float, scale_max: float, sigma: float) -> ndarray:
    m_max = np.floor(np.log(scale_min, scale_max) / np.log(sigma))
    m = np.arange(0, m_max + 2)

    scalefreq = np.empty((len(m),))

    for z in range(len(m)):
        scalefreq[z] = 1 / (scale_min * (1.05 ** (z - 1)))

    return scalefreq


def modbasicwavelet_flow_cmplx4(
    signal: ndarray,
    fs: float,
    scale_min: float,
    scale_max: float,
    sigma: float,
    time_res: float,
) -> ndarray:
    t_start = 0
    t_end = len(signal) / fs

    m_max = np.floor(np.log(scale_max / scale_min) / np.log(sigma))
    m = np.arange(0, m_max + 2, 1)

    REZ = np.empty((len(m), np.floor(t_end - t_start) / time_res + 1)).fill(np.NaN)
    flo = np.floor((t_end - t_start))
    stevec = 0

    for z in range(1, len(m)):
        s = scale_min * sigma ** m[z]

        # Begin calculating wavelet.
        f0 = 1

        tval_k = 0
        tval_k2 = 0

        for ttt in np.arange(0, 5001, 1 / fs):
            zzz = s ** (-0.5) * np.exp(-(ttt ** 2) / (2 * s ** 2))

            if zzz < 0.1 * (s ** (-0.5)) and tval_k2 == 0:
                tval_k2 = ttt
                # check1[z] = ttt

            if zzz < 0.001 * (s ** (-0.5)):
                tval_k = ttt
                # check2[z] = ttt

        st_kor = tval_k * fs
        margin = tval_k2 * fs

        # Round up st_kor for accuracy.
        if np.mod(st_kor, fs) != 0:
            st_kor = fs * np.ceil(st_kor / fs)

        # Round up margin for safety.
        if np.mod(margin, fs) != 0:
            margin = fs * np.ceil(margin / fs)

        u = np.arange(-st_kor / fs, st_kor / fs, 1 / fs)

        wavelet = (
            s ** (-0.5)
            * np.exp(-2j * np.pi * f0 * u / s)
            * np.exp(-(u ** 2) / (2 * s ** 2))
        )

        X = np.fft.fft(np.concatenate((wavelet, np.zeros(1, len(signal)))), axis=0)
        Y = np.fft.fft(np.concatenate((signal, np.zeros(1, len(wavelet)))), axis=0)

        con = np.fft.ifft(X * Y, axis=0)

        rez = con[
            np.arange(
                st_kor + margin, (st_kor - margin + (flo * fs) + 1), fs * time_res
            )
        ]

        stevec += 1
        if margin / (fs * time_res) > 0:
            trans = np.concatenate(
                (
                    np.empty((1, margin / (fs * time_res))).fill(np.NaN),
                    rez,
                    np.empty((1, margin / (fs * time_res))),
                )
            )
        else:
            trans = rez.copy()

        lentrans = len(trans)
        lenrez = len(REZ[stevec, :])

        if lentrans == lenrez:
            REZ[stevec, :] = trans

    return REZ


def indexfinder3(data1, data2) -> Tuple[ndarray, ndarray]:
    bins = 24

    i1 = np.argsort(data1)
    y1 = np.sort(data1)

    i2 = np.argsort(data2)
    y1 = np.sort(data2)

    dummy1 = np.arange(0, len(data1))
    dummy2 = np.arange(0, len(data2))

    pslow[i1] = np.ceil(bins * dummy1 / len(data1))
    pfast[i1] = np.ceil(bins * dummy2 / len(data2))

    binner = np.zeros((bins, bins,))
    for n in range(len(pslow)):
        binner[pslow[n], pfast[n]] = binner[pslow[n], pfast[n]] + 1

    total = np.sum(np.sum(binner))
    if total == 0:
        meanindex = np.NaN
    else:
        i2 = 0

        for n in range(bins):
            i1[n] = 0

            for m in range(bins):
                if np.sum(binner[n, :]) != 0:
                    pa = binner[n, m] / np.sum(binner[n, :])
                else:
                    pa = 0

                if pa != 0:
                    i1[n] = i1[n] - pa * np.log2(pa)

            pb = np.sum(binner[n, :]) / total
            i2 += pb * i1[n]

        i3 = 0
        for n in range(bins):
            if np.sum(binner[:, n]):
                pc = np.sum(binner[:, n]) / np.sum(np.sum(binner))
            else:
                pc = 0

            if pc != 0:
                i3 -= pc * np.log2(pc)

        meanindex = (i3 - i2) / i3

    return binner, meanindex
