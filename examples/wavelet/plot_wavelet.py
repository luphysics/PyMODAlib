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

import pymodalib

os.chdir(os.path.abspath(os.path.dirname(__file__)))

print("Plotting...")

assert "output.npz" in os.listdir("."), (
    "Please run 'wavelet-python.py' or 'wavelet-matlab.py' to generate "
    "the data required for this script to run."
)
load = np.load("output.npz")

wt = load.get("wt")
freq = load.get("freq")
times = load.get("times")
impl = load.get("implementation")

wt_power = np.abs(wt) ** 2
avg_wt_power = np.nanmean(wt_power, axis=1)
mesh1, mesh2 = np.meshgrid(times, freq)

fig, (ax1, ax2) = plt.subplots(
    1,
    2,
    gridspec_kw={
        "width_ratios": [
            3,
            1,
        ]
    },
    sharey=True,
)

pymodalib.contourf(ax1, mesh1, mesh2, wt_power)
ax1.set_title(f"WT power ({impl} implementation)")
ax1.set_xlabel("Time (s)")

ax2.plot(avg_wt_power, freq)
ax2.set_title("Time-averaged WT power")

for ax in (ax1, ax2):
    ax.set_yscale("log")
    ax.set_ylabel("Frequency (Hz)")

ax1.set_ylim(np.min(freq), np.max(freq))

fig.set_size_inches(10, 5.5)
plt.tight_layout()
plt.show()
