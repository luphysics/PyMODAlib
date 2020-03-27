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
import os
import platform

import numpy
import numpy as np

from pymodalib.algorithms.wavelet import wavelet_transform

os.chdir(os.path.abspath(os.path.dirname(__file__)))

if __name__ == "__main__":
    signal = np.load("1signal_10Hz.npy")

    fs = 10
    times = np.arange(0, signal.size / fs, 1 / fs)

    wt, freq = wavelet_transform(signal, fs, cut_edges=True, return_opt=False)
    numpy.savez("output", wt=wt, freq=freq, times=times, implementation="MATLAB")

    if platform.system() != "Linux":
        import plot_wavelet

        dummy_variable = plot_wavelet
    else:
        print(
            "\nResults saved to a data file. Please run the 'plot_wavelet.py' script to plot the results."
        )
