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

from typing import Tuple, Union, Dict

import numpy as np
from numpy import ndarray

from pymodalib.utils.decorators import matlabwrapper
from pymodalib.utils.parameters import sanitise


@matlabwrapper(module="ecurve")
def ecurve_impl(
    tfr: ndarray,
    frequencies: ndarray,
    fs: float,
    method: Union[int, str, ndarray] = None,
    param: Union[float, Tuple[float, float]] = None,
    normalize: bool = None,
    wopt: Dict = None,
    path_optimize: bool = None,
    max_iterations: int = None,
    **kwargs,
) -> ndarray:
    import ecurve
    import matlab

    package = ecurve.initialize()

    # We only need absolute values.
    tfr = matlab.double(np.abs(tfr).tolist())
    frequencies = matlab.double(frequencies.tolist())

    wopt = wopt or {}
    wopt["fs"] = float(fs)

    kwargs = sanitise(
        {
            "method": method,
            "Normalize": "on" if normalize else "off",
            "PathOpt": "on" if path_optimize else "off",
            "MaxIter": max_iterations,
            "Param": param,
            **kwargs,
        }
    )

    tfsupp = package.ecurve(tfr, frequencies, wopt, kwargs, nargout=1)
    return tfsupp


@matlabwrapper(module="rectfr")
def rectfr_impl(
    tfsupp: ndarray,
    tfr: ndarray,
    frequencies: ndarray,
    fs: float,
    wopt: Dict = None,
    method: str = "direct",
) -> Tuple:
    import rectfr
    import matlab

    package = rectfr.initialize()

    if isinstance(tfsupp, ndarray):
        tfsupp = matlab.double(tfsupp.tolist())

    args = [
        method,
    ]

    if isinstance(tfr, ndarray):
        tfr_real = matlab.double(np.real(tfr).tolist())
        tfr_imag = matlab.double(np.imag(tfr).tolist())

        tfr = tfr_real
        args.append(tfr_imag)

    if isinstance(frequencies, ndarray):
        frequencies = matlab.double(frequencies.tolist())

    wopt = wopt or {}
    wopt["fs"] = float(fs)

    iamp, iphi, ifreq = package.rectfr(tfsupp, tfr, frequencies, wopt, *args, nargout=3)
    return iamp, iphi, ifreq
