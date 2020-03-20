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
This is for testing the group coherence algorithm with known data.

It may be useful to read the code, but the data are not supplied as part of PyMODAlib.
"""

import os

import numpy as np
import scipy.io
from matplotlib import pyplot as plt
from numpy import ndarray

from pymodalib.algorithms.group_coherence import dual_group_coherence

fs = 31.25

minutes = 1
sig_length = int(fs * 60 * minutes)


def load_mat(filename: str) -> ndarray:
    cell = list(scipy.io.loadmat(filename).values())[3]

    out = np.empty((cell.shape[1], sig_length))

    for index in range(cell.shape[1]):
        out[index, :] = cell[0, index][0, :sig_length]

    return out


if __name__ == "__main__":
    os.chdir(os.path.abspath(os.path.dirname(__file__)))

    times = np.arange(0, sig_length / fs, 1 / fs)

    group1_signals_a = load_mat("Cphd_O2satNIRS11.mat")
    group1_signals_b = load_mat("Cphd_Respiration_resampl.mat")
    group2_signals_a = load_mat("phd_O2satNIRS11.mat")
    group2_signals_b = load_mat("phd_Respiration_resampl.mat")

    results = dual_group_coherence(
        group1_signals_a, group1_signals_b, group2_signals_a, group2_signals_b, fs
    )

    freq, coh1, coh2, surr1, surr2 = results

    median1 = np.median(coh1, axis=0)
    median2 = np.median(coh2, axis=0)

    pc1_25 = np.percentile(coh1, 25, axis=0)
    pc1_75 = np.percentile(coh1, 75, axis=0)
    pc2_25 = np.percentile(coh2, 25, axis=0)
    pc2_75 = np.percentile(coh2, 75, axis=0)

    colour1 = "black"
    colour2 = "red"

    plt.plot(freq, median1, color=colour1)
    plt.plot(freq, median2, color=colour2)

    alpha = 0.1
    plt.fill_between(freq, pc1_25, pc1_75, color=colour1, alpha=alpha)
    plt.fill_between(freq, pc2_25, pc2_75, color=colour2, alpha=alpha)

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Coherence")

    plt.gca().set_xscale("log")

    plt.legend(["Median coherence (group 1)", "Median coherence (group 2)"])
    plt.show()
