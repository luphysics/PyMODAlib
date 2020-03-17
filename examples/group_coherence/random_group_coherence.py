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
import random

import numpy as np
from matplotlib import pyplot

from pymodalib.algorithms.group_coherence import dual_group_coherence


def generate_signal(times):
    f = random.randint(1, 10)

    return np.sin(f * times)


if __name__ == "__main__":
    import os
    import platform

    if platform.system() == "Linux":
        os.environ["LD_LIBRARY_PATH"] = (
            "/usr/local/MATLAB/MATLAB_Runtime/v96/runtime/glnxa64:"
            "/usr/local/MATLAB/MATLAB_Runtime/v96/bin/glnxa64:"
            "/usr/local/MATLAB/MATLAB_Runtime/v96/sys/os/glnxa64:"
            "/usr/local/MATLAB/MATLAB_Runtime/v96/extern/bin/glnxa64"
        )

    fs = 10
    times = np.arange(0, 1000 / fs, 1 / fs)

    num_signals = 20

    group1_signals_a = np.asarray([generate_signal(times) for _ in range(num_signals)])
    group1_signals_b = np.asarray([generate_signal(times) for _ in range(num_signals)])

    group2_signals_a = np.asarray([generate_signal(times) for _ in range(num_signals)])
    group2_signals_b = np.asarray([generate_signal(times) for _ in range(num_signals)])

    results = dual_group_coherence(
        group1_signals_a, group1_signals_b, group2_signals_a, group2_signals_b, fs
    )

    freq, coh1, _, coh2 = results

    mean1 = np.nanmean(coh1, axis=0)
    mean2 = np.nanmean(coh2, axis=0)

    pyplot.plot(freq, mean1)
    pyplot.plot(freq, mean2)

    pyplot.legend(["Mean coherence (group 1)", "Mean coherence (group 2)"])
    pyplot.show()
