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
from typing import Tuple, Dict

from numpy import ndarray

from pymodalib.utils.decorators import matlabwrapper
from pymodalib.utils.matlab import matlab_to_numpy
from pymodalib.utils.parameters import sanitise, float_or_none


@matlabwrapper
def wavelet_transform(
    signal: ndarray,
    fs: float,
    fmin: float = None,
    fmax: float = None,
    resolution: float = 1,
    cut_edges: bool = False,
    wavelet: str = "Lognorm",
    preprocess: bool = True,
    padding: str = "predictive",
    fstep: str = "auto",
    rel_tolerance: float = 0.01,
) -> Tuple[ndarray, ndarray, Dict]:
    """
    MATLAB implementation of the wavelet transform.
    """

    kwargs = sanitise(
        {
            "fmin": float_or_none(fmin),
            "fmax": float_or_none(fmax) if fmax else fs / 2.0,
            "f0": float_or_none(resolution),
            "CutEdges": "on" if cut_edges else "off",
            "Padding": padding,
            "fstep": fstep if isinstance(fstep, str) else float_or_none(fstep),
            "RelTol": float_or_none(rel_tolerance),
            "Wavelet": wavelet,
            "Preprocess": "on" if preprocess else "off",
            "python": True,
        }
    )

    import WT
    import matlab

    package = WT.initialize()

    wt, freq, opt = package.wt(
        matlab.double(signal.tolist()), float(fs), kwargs, nargout=3
    )
    return matlab_to_numpy(wt), matlab_to_numpy(freq), opt
