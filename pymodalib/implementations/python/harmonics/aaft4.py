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
Python implementation of the AAFT surrogate from MATLAB code, `aaft4.m`, supplied by Lawrence Sheppard.
"""

from typing import Tuple

import numpy as np
from numpy import ndarray


def aaft4(seriesa: ndarray) -> Tuple[ndarray, ndarray]:
    try:
        idim, jdim = seriesa.shape
    except ValueError:
        (idim,) = seriesa.shape

    if idim > 1:
        seriesa = seriesa.conj().T

    np.random.rand()

    amplitudes = np.random.randn(1, len(seriesa))

    Y1 = np.sort(amplitudes)
    Y2 = np.sort(seriesa)
    I2 = np.argsort(seriesa)

    seriesc = np.empty(seriesa.shape)
    seriesc[I2] = Y1

    fftseries = np.fft.fft(seriesc, axis=0)

    if np.mod(len(seriesa), 2) == 1:
        numbers = np.random.rand(1, (len(fftseries) - 1) / 2)
        phases = 2 * np.pi * numbers
        phasors = np.exp(1j * phases)

        rotfftseries = fftseries * np.concatenate(
            (np.ones((1,)), phasors, np.ones((1,)), np.fliplr(phasors.conj()))
        )

        ifftrot = np.fft.ifft(rotfftseries, axis=0)

        I3 = np.argsort(ifftrot)
        output = np.empty(ifftrot.shape)
        output[I3] = Y2
    else:
        numbers = np.random.rand(1, len(fftseries) // 2 - 1)

        phases = 2 * np.pi * numbers
        phasors = np.exp(1j * phases)

        rotfftseries = fftseries * np.concatenate(
            (
                np.ones((1,)),
                phasors[0, :],
                np.ones((1,)),
                np.fliplr(phasors.conj())[0, :],
            )
        )

        ifftrot = np.fft.ifft(rotfftseries, axis=0)

        I3 = np.argsort(ifftrot)
        output = np.empty(ifftrot.shape)
        output[I3] = Y2

    origorder = np.arange(0, len(seriesa))
    sortedorder = origorder[I2]

    finalorder = np.empty(sortedorder.shape)
    finalorder[I3] = sortedorder

    return output, finalorder
