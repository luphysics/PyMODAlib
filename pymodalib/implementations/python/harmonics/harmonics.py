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

from typing import Tuple, List

import numpy as np
from numpy import ndarray
from scheduler.Scheduler import Scheduler

from pymodalib.implementations.python.harmonics.aaft4 import aaft4


def harmonicfinder_impl_python(
    signal: ndarray,
    fs: float,
    scale_min: float,
    scale_max: float,
    sigma: float,
    time_res: float,
    surr_count: int,
) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    """
    Python implementation of the harmonicfinder function.
    """
    try:
        x, y = signal.shape
        if y > x:
            signal = signal[0, :]
    except ValueError:
        pass

    output1 = modbasicwavelet_flow_cmplx4(
        signal, fs, scale_min, scale_max, sigma, time_res
    )

    scalefreq = scale_frequency(scale_min, scale_max, sigma)

    m, n = output1.shape
    detsig = signal

    ressur = np.empty((surr_count, m, m,))
    ressur.fill(np.NaN)

    scheduler = Scheduler(shared_memory=False)

    surr_args = [
        (i, output1, m, n, detsig, fs, scale_min, scale_max, sigma, time_res)
        for i in range(surr_count)
    ]

    # Add surrogate calculations to scheduler.
    for args in surr_args:
        scheduler.add(target=_calculate_surrogate, args=args)

    if surr_count > 0:
        # Add harmonic calculation for main signal to scheduler. This should be the last item added.
        scheduler.add(target=_calculate_harmonics, args=(output1, m, n))

        # Run scheduler.
        result: List[ndarray] = scheduler.run_blocking()

        # Get main harmonics result.
        res = result[-1]

        # Get surrogate results.
        for sigb in range(surr_count):
            ressur[sigb, :, :] = result[sigb][:, :]
    else:
        res = _calculate_harmonics(output1, m, n, parallel=False)

    sig = np.empty((1 + ressur.shape[0],))
    pos = np.empty((m, m))

    for a1 in range(m):
        for a2 in range(1, a1 + 2):
            _res_a1_a2 = res[a1, a2 - 1]
            if not hasattr(_res_a1_a2, "__len__"):
                _res_a1_a2 = np.asarray((_res_a1_a2,))

            isurr = np.argsort(np.concatenate((_res_a1_a2, ressur[:, a1, a2 - 1],)))
            sig[isurr] = np.arange(0, surr_count + 1)

            pos[a1, a2 - 1] = sig[0]
            pos[a2 - 1, a1] = sig[0]

    pos1 = pos.copy()

    surrmean = np.empty((m, m,))
    surrstd = np.empty((m, m,))

    for a1 in range(m):
        for a2 in range(m):
            surrmean[a1, a2] = np.nanmean(ressur[:, a1, a2])
            surrstd[a1, a2] = np.nanstd(ressur[:, a1, a2])
            sig = (res[a1, a2] - surrmean[a1, a2]) / surrstd[a1, a2]
            pos[a1, a2] = np.min([sig, 5])

    pos2 = pos

    return (
        scalefreq,
        res.conj().T,
        pos1.conj().T,
        pos2.conj().T,
    )


def _calculate_harmonics(output1, m, n, parallel=True) -> ndarray:
    print(f"Calculating harmonics{' (running in parallel)' if parallel else ''}...")

    res = np.empty((m, m,))
    res.fill(np.NaN)

    for a1 in range(m):
        margin = np.int(np.ceil((np.sum(np.isnan(np.angle(output1[a1, : n + 1])))) / 2))
        phase1 = np.angle(output1[a1, margin : n - margin])  # Slow.

        for a2 in range(1, a1 + 2):
            phase2 = np.angle(output1[a2 - 1, margin : n - margin])  # Fast.

            _, mean_index = indexfinder3(phase1, phase2)
            res[a1, a2 - 1] = mean_index
            res[a2 - 1, a1] = mean_index

    return res


def _calculate_surrogate(
    sigb, output1, m, n, detsig, fs, scale_min, scale_max, sigma, time_res
) -> ndarray:
    print(f"Calculating surrogate {sigb + 1} (running in parallel)...")

    ressur = np.empty((m, m,))
    ressur.fill(np.NaN)

    # This is important. When running in parallel, the random state is identical for all processes;
    # rand() needs to be called according to the index of the process (which is given by the surrogate number, 'sigb').
    for i in range(sigb):
        np.random.rand()

    surrsig, _ = aaft4(detsig.conj().T)
    transsurr = modbasicwavelet_flow_cmplx4(
        surrsig, fs, scale_min, scale_max, sigma, time_res
    )

    for a1 in range(m):
        margin = np.int(np.ceil((np.sum(np.isnan(np.angle(output1[a1, : n + 1])))) / 2))
        phase1 = np.angle(output1[a1, margin : n - margin])  # Slow.

        for a2 in range(1, a1 + 2):
            phase2 = np.angle(transsurr[a2 - 1, margin : n - margin])  # Fast.

            _, mean_index = indexfinder3(phase1, phase2)
            ressur[a1, a2 - 1] = mean_index
            ressur[a2 - 1, a1] = mean_index

    return ressur


def scale_frequency(scale_min: float, scale_max: float, sigma: float) -> ndarray:
    m_max = np.floor(np.log(scale_max / scale_min) / np.log(sigma))
    m_max = np.int(m_max)

    m = np.arange(0, m_max + 2)

    scalefreq = np.empty((len(m),))

    for z in range(len(m)):
        scalefreq[z] = 1 / (scale_min * (sigma ** z))

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

    m_max = np.int(np.floor(np.log(scale_max / scale_min) / np.log(sigma)))
    m = np.arange(0, m_max + 2)

    REZ = np.empty(
        (len(m), np.int(np.floor(t_end - t_start) / time_res + 1)), dtype=np.complex64
    )
    REZ.fill(np.NaN)

    flo = np.int(np.floor((t_end - t_start)))
    stevec = -1

    for z in range(len(m)):
        s = scale_min * sigma ** m[z]

        # Begin calculating wavelet.
        f0 = 1

        tval_k = 0
        tval_k2 = 0

        _ttt = np.arange(0, 5000 + 0.5 / fs, 1 / fs)
        _zzz = s ** (-0.5) * np.exp(-(_ttt ** 2) / (2 * s ** 2))

        for index in range(len(_zzz)):
            zzz = _zzz[index]
            ttt = _ttt[index]

            if zzz < 0.1 * (s ** (-0.5)) and tval_k2 == 0:
                tval_k2 = ttt

            if zzz < 0.001 * (s ** (-0.5)):
                tval_k = ttt
                break

        st_kor = np.int(tval_k * fs)
        margin = tval_k2 * fs

        # Round up st_kor for accuracy.
        if np.mod(st_kor, fs) != 0:
            st_kor = np.int(fs * np.ceil(st_kor / fs))

        # Round up margin for safety.
        if np.mod(margin, fs) != 0:
            margin = np.int(fs * np.ceil(margin / fs))

        if margin > 0.5 * fs * flo:
            break

        u = np.arange(-st_kor / fs, st_kor / fs, 1 / fs)

        wavelet = (
            s ** (-0.5)
            * np.exp(-2j * np.pi * f0 * u / s)
            * np.exp(-(u ** 2) / (2 * s ** 2))
        )

        x = np.fft.fft(np.concatenate((wavelet, np.zeros((len(signal) - 1,)))), axis=0)
        y = np.fft.fft(np.concatenate((signal, np.zeros((len(wavelet) - 1,)))), axis=0)

        con = np.fft.ifft(x * y, axis=0)

        step = np.int(fs * time_res)
        if step != fs * time_res:
            raise Exception("'fs' and 'time_res' must be integers.")

        margin = np.int(margin)
        rez = con[np.arange(st_kor + margin, (st_kor - margin + (flo * fs) + 1), step)]

        stevec += 1
        if margin / (fs * time_res) > 0:
            cols = np.int(margin / (fs * time_res))
            nan = np.empty((cols,))
            nan.fill(np.NaN)
            trans = np.concatenate((nan, rez, nan,))
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
    i2 = np.argsort(data2)

    # y1 = np.sort(data1)
    # y1 = np.sort(data2)

    dummy1 = np.arange(0, len(data1))
    dummy2 = np.arange(0, len(data2))

    pslow = np.empty(dummy1.shape, dtype=np.int64)
    pfast = np.empty(dummy2.shape, dtype=np.int64)

    pslow[i1] = np.floor(bins * dummy1 / len(data1))
    pfast[i2] = np.floor(bins * dummy2 / len(data2))

    binner = np.zeros((bins, bins,))

    for n in range(len(pslow)):
        binner[pslow[n], pfast[n]] = binner[pslow[n], pfast[n]] + 1

    total = np.sum(np.sum(binner))
    if total == 0:
        meanindex = np.NaN
    else:
        i2 = 0

        for n in range(bins):
            i1n = 0

            for m in range(bins):
                if np.sum(binner[n, :]) != 0:
                    pa = binner[n, m] / np.sum(binner[n, :])
                else:
                    pa = 0

                if pa != 0:
                    i1n = i1n - pa * np.log2(pa)

            pb = np.sum(binner[n, :]) / total
            i2 += pb * i1n

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
