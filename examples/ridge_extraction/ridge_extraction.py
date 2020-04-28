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
Example of using the ridge extraction algorithm.
"""
import os

import numpy as np
from matplotlib import pyplot as plt

import pymodalib

os.chdir(os.path.abspath(os.path.dirname(__file__)))

if __name__ == "__main__":
    signal = np.load("../1signal_10Hz.npy")

    fs = 10
    times = np.arange(0, signal.size / fs, 1 / fs)

    # Frequency interval (0.07Hz-0.33Hz).
    fmin, fmax = 0.07, 0.33

    # Note: your IDE may incorrectly display an error on this statement.
    wt, freq, wopt = pymodalib.wavelet_transform(
        signal,
        fs,
        fmin=fmin,
        fmax=fmax,
        cut_edges=True,
        return_opt=True,
        implementation="matlab",
    )

    iamp, iphi, ifreq = pymodalib.ridge_extraction(wt, freq, fs, wopt=wopt)

    ax1 = plt.subplot(311)
    ax1.plot(times, iamp)

    ax2 = plt.subplot(312)
    ax2.plot(times, iphi)
    ax2.pco

    ax3 = plt.subplot(313)
    ax3.plot(times, ifreq)

    plt.show()
