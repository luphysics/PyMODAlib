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
from typing import Tuple

from numpy import ndarray

from pymodalib.utils.decorators import matlabwrapper
from pymodalib.utils.matlab import matlab_to_numpy
from pymodalib.utils.parameters import sanitise, float_or_none


@matlabwrapper(module="WFT")
def wft_impl(
    signal,
    fs,
    fmin,
    fmax,
    resolution,
    cut_edges,
    window,
    preprocess,
    padding,
    fstep,
    rel_tolerance,
    *args,
    **kwargs,
) -> Tuple[ndarray, ndarray]:
    import WFT
    import matlab

    kwargs = sanitise(
        {
            "fmin": float_or_none(fmin),
            "fmax": float_or_none(fmax),
            "f0": float_or_none(resolution),
            "CutEdges": "on" if cut_edges else "off",
            "Padding": padding,
            "fstep": fstep,
            "RelTol": float_or_none(rel_tolerance),
            "Window": window,
            "Preprocess": "on" if preprocess else "off",
            **kwargs,
        }
    )

    package = WFT.initialize()

    wft, freq = package.wft(
        matlab.double(signal.tolist()), float(fs), kwargs, nargout=2
    )
    return matlab_to_numpy(wft), matlab_to_numpy(freq)
