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
import sys

import numpy as np
import scipy.io
from matplotlib import pyplot as plt
from numpy import ndarray

from pymodalib.algorithms.group_coherence import dual_group_coherence

# Sampling frequency.
fs = 31.25

# Length of the signals in minutes. We'll trim them down to this length when they're loaded.
minutes = 1
sig_length = int(fs * 60 * minutes)


def load_mat(filename: str) -> ndarray:
    """
    Loads a signal from one of the .mat files.
    """
    cell = list(scipy.io.loadmat(filename).values())[3]
    out = np.empty((cell.shape[1], sig_length))

    for index in range(cell.shape[1]):
        out[index, :] = cell[0, index][0, :sig_length]

    return out


if __name__ == "__main__":
    # Change working directory to where this file is saved.
    os.chdir(os.path.abspath(os.path.dirname(__file__)))

    # Create array of time values.
    times = np.arange(0, sig_length / fs, 1 / fs)

    try:
        # Load the data files.
        group1_signals_a = load_mat("Cphd_O2satNIRS11.mat")
        group1_signals_b = load_mat("Cphd_Respiration_resampl.mat")
        group2_signals_a = load_mat("phd_O2satNIRS11.mat")
        group2_signals_b = load_mat("phd_Respiration_resampl.mat")
    except:
        print(
            "Unfortunately, you don't seem to have the data files required to run this analysis."
        )
        sys.exit(-1)

    # Calculate the group coherence.
    freq, coh1, coh2, surr1, surr2 = dual_group_coherence(
        group1_signals_a, group1_signals_b, group2_signals_a, group2_signals_b, fs
    )

    # Save the results as a data file.
    np.savez(f"output.npz", freq=freq, coh1=coh1, coh2=coh2)

    # Calculate median coherence.
    median1 = np.nanmedian(coh1, axis=0)
    median2 = np.nanmedian(coh2, axis=0)

    # Calculate 25th and 75th percentiles of the surrogates.
    pc1_25 = np.nanpercentile(coh1, 25, axis=0)
    pc1_75 = np.nanpercentile(coh1, 75, axis=0)
    pc2_25 = np.nanpercentile(coh2, 25, axis=0)
    pc2_75 = np.nanpercentile(coh2, 75, axis=0)

    colour1 = "black"
    colour2 = "red"

    # Plot the median coherences.
    plt.plot(freq, median1, color=colour1)
    plt.plot(freq, median2, color=colour2)

    # Plot the filled areas between percentiles.
    alpha = 0.1
    plt.fill_between(freq, pc1_25, pc1_75, color=colour1, alpha=alpha)
    plt.fill_between(freq, pc2_25, pc2_75, color=colour2, alpha=alpha)

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Coherence")
    plt.gca().set_xscale("log")

    plt.legend(["Median coherence (group 1)", "Median coherence (group 2)"])
    plt.show()
