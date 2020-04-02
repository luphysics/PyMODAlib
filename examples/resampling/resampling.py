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
from numpy import ndarray

import pymodalib

os.chdir(os.path.abspath(os.path.dirname(__file__)))

signal = np.load("../1signal_10Hz.npy").transpose()
fs = 10


def times(signal, fs) -> ndarray:
    return np.arange(0, len(signal) / fs, 1 / fs)


fs1 = 1
resampl_1 = pymodalib.downsample(signal, fs, fs1)

fs2 = 0.2
resampl_2 = pymodalib.downsample(signal, fs, fs2)

ax1 = plt.subplot(3, 1, 1)
ax1.plot(times(signal, fs), signal)
ax1.set_title("Original signal")

ax2 = plt.subplot(3, 1, 2)
ax2.plot(times(resampl_1, fs1), resampl_1)
ax2.set_title(f"Signal downsampled to {fs1}Hz")

ax3 = plt.subplot(3, 1, 3)
ax3.plot(times(resampl_2, fs2), resampl_2)
ax3.set_title(f"Signal downsampled to {fs2}Hz")

plt.tight_layout()
plt.show()
