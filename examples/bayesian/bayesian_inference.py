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
import chdir

# Set working directory to location of current file.
from scipy.io import loadmat

import pymodalib

chdir.here(__file__)

if __name__ == "__main__":
    signals = loadmat("../2signals_10Hz.mat")["a"]

    sig1 = signals[0, :]
    sig2 = signals[1, :]

    fs = 10

    interval1 = [0.081, 0.3]
    interval2 = [0.08, 0.31]

    result = pymodalib.bayesian_inference(
        sig1,
        sig2,
        fs,
        interval1,
        interval2,
        surrogates=4,
        window=50,
        overlap=1,
        order=2,
        propagation_const=0.2,
        signif=0.95,
    )
