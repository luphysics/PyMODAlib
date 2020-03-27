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

os.chdir(os.path.abspath(os.path.dirname(__file__)))

load = np.load("output.npz")

wt = load.get("wt")
freq = load.get("freq")
times = load.get("times")
impl = load.get("implementation")

fig, ax = plt.subplots()
mesh1, mesh2 = np.meshgrid(times, freq)
ax.contourf(mesh1, mesh2, np.abs(wt))

ax.set_yscale("log")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Frequency (Hz)")
ax.set_title(f"Amplitude of wavelet transform ({impl} implementation)")

plt.show()
