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
This script uses the result of "nirs_group_coherence.py" to perform a statistical test
for the significance of the results.
"""

import os

import numpy as np
from matplotlib import pyplot as plt

from pymodalib.algorithms import group_coherence

if __name__ == "__main__":
    # Change working directory to where this file is saved.
    os.chdir(os.path.abspath(os.path.dirname(__file__)))

    data = np.load("output.npz", allow_pickle=True)
    freq = data.get("freq")
    coh1 = data.get("coh1")
    coh2 = data.get("coh2")

    intervals = {
        "Endothelial": (0.005, 0.021),
        "Neurogenic": (0.021, 0.052),
        "Myogenic": (0.052, 0.145),
        "Respiration": (0.145, 0.6),
        "Cardiac": (0.6, 2),
    }

    pvalues = group_coherence.statistical_test(freq, coh1, coh2, intervals.values())

    for index, name in enumerate(intervals):
        print(f"{name}: {pvalues[index]}")

    significance = 5 / 100
    text = [
        [name, intervals[name], pvalues[index]] for index, name in enumerate(intervals)
    ]

    plt.table(text)
