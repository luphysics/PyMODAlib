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
import random

import numpy as np
from matplotlib import pyplot as plt

from pymodalib.algorithms.group_coherence import dual_group_coherence

"""
This is an example of running the group coherence algorithm for 2 groups. Please note that the signals used are 
randomly generated, so the coherence will essentially be zero. 

The results are not useful, but it provides an example of how to use the algorithm and a way to test the 
algorithm on your system.
"""


def generate_signal(times):
    f = random.random()

    return np.sin(f * times) + random.randint(1, 100)


if __name__ == "__main__":
    os.chdir(os.path.abspath(os.path.dirname(__file__)))

    fs = 10
    times = np.arange(0, 1000 / fs, 1 / fs)

    num_signals = 20

    # Generate random signals. Note: random signals have almost zero coherence, so the results will be somewhat useless.
    group1_signals_a = np.asarray([generate_signal(times) for _ in range(num_signals)])
    group1_signals_b = np.asarray([generate_signal(times) for _ in range(num_signals)])
    group2_signals_a = np.asarray([generate_signal(times) for _ in range(num_signals)])
    group2_signals_b = np.asarray([generate_signal(times) for _ in range(num_signals)])

    # Calculate group coherence.
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
