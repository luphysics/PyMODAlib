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
Example of using the ridge extraction algorithm. Calculates and plots the frequency ridge.
"""
import os

import numpy as np
from matplotlib import pyplot as plt

import pymodalib

os.chdir(os.path.abspath(os.path.dirname(__file__)))

from pymodalib.utils import macos

macos.configure_dyld_library_path(
    "/Applications/MATLAB/MATLAB_Runtime/v96/runtime/maci64:"
    "/Applications/MATLAB/MATLAB_Runtime/v96/sys/os/maci64:"
    "/Applications/MATLAB/MATLAB_Runtime/v96/bin/maci64:"
    "/Applications/MATLAB/MATLAB_Runtime/v96/extern/bin/maci64"
)

if __name__ == "__main__":
    signal = np.load("../1signal_10Hz.npy")

    fs = 10

    # Time values associated with the signal.
    times = np.arange(0, signal.size / fs, 1 / fs)

    # Frequency interval (0.07Hz-0.33Hz).
    fmin, fmax = 0.07, 0.33

    # Note: your IDE may incorrectly display an error on this statement.
    wt, freq, wopt = pymodalib.wavelet_transform(
        signal,
        fs,
        fmin=fmin,
        fmax=fmax,
        return_opt=True,
        implementation="matlab",
    )

    iamp, iphi, ifreq = pymodalib.ridge_extraction(wt, freq, fs, wopt=wopt)

    fig, ax = plt.subplots()

    wt_amp = np.abs(wt)
    mesh1, mesh2 = np.meshgrid(times, freq)

    # Plot wavelet transform.
    ax.contourf(mesh1, mesh2, wt_amp)

    # Plot frequency ridge.
    ax.plot(times, ifreq, color="red")

    ax.set_title("Frequency ridge and wavelet transform")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_yscale("log")

    fig.set_size_inches(9, 7)
    plt.show()
