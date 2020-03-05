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
import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat

from pymodalib.algorithms.harmonics.harmonics import harmonicfinder


def test_harmonics():
    fs = 50

    # time = 100
    # freq = 1 / 7
    # times = np.arange(1 / fs, time + 1 / fs, 1 / fs)
    # signal = (
    #                  0.1 * np.random.randn(1, len(times)) + np.sin(2 * np.pi * freq * times) ** 3
    #          )[0, :]

    signal = loadmat("t_series.mat").get("t_series")[0, :]

    scale_min = 0.5
    scale_max = 40
    sigma = 1.05
    time_res = 0.1

    scale_freq, res, pos1, pos2 = harmonicfinder(
        signal, fs, scale_min, scale_max, sigma, time_res, 2
    )

    mesh_plot(scale_freq, scale_freq, res, "Raw harmonics")
    mesh_plot(scale_freq, scale_freq, pos1, "Higher than how many AAFT surrogates")
    mesh_plot(
        scale_freq,
        scale_freq,
        pos2,
        "Relative to mean and std of surrogate distribution",
    )

    plt.show()


def mesh_plot(x, y, c, title):
    fig, ax = plt.subplots()

    ax.pcolormesh(x, y, c, shading="flat")

    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_title(title)


if __name__ == "__main__":
    test_harmonics()
