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

import numpy as np
from matplotlib import pyplot as plt

from pymodalib import preprocess

# Set current working directory to the location of this script.
os.chdir(os.path.abspath(os.path.dirname(__file__)))

signal = np.load("1signal_10Hz.npy")

# Sampling frequency.
fs = 10

# Times associated with signal.
times = np.arange(0, signal.size / fs, 1 / fs)

preproc_signal = preprocess(signal, fs)

ax1 = plt.subplot(2, 1, 1)
ax1.plot(times, signal[0, :])

ax2 = plt.subplot(2, 1, 2, sharex=ax1, sharey=ax1)
ax2.plot(times, preproc_signal)

plt.show()
