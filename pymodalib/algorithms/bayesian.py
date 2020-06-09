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
from typing import Any, Iterable

from numpy import ndarray

from pymodalib.utils.decorators import experimental


def bayesian_inference(
    signal1: ndarray,
    signal2: ndarray,
    fs: float,
    interval1: Iterable[float],
    interval2: Iterable[float],
    surrogates: int,
    window: float,
    overlap: float,
    propagation_const: float,
    order: int,
    signif: float,
    implementation="matlab",
    *args,
    **kwargs,
) -> Any:
    """
    Dynamical Bayesian inference.

    Parameters
    ----------
    signal1 : ndarray
        The first signal.
    signal2
    fs
    interval1
    interval2
    surrogates
    window
    overlap
    propagation_const
    order
    signif
    implementation : {"matlab", "python"}
        (Default = "matlab")

    Returns
    -------
    tm
    p1
    p2
    cpl1
    cpl2
    cf1
    cf2
    mcf1
    mcf2
    surr_cpl1
    surr_cpl2
    """
    from pymodalib.implementations.python.bayesian import (
        bayesian_inference_impl as python_impl,
    )
    from pymodalib.implementations.matlab.bayesian import (
        bayesian_inference_impl as matlab_impl,
    )

    if implementation == "python":
        result = python_impl(
            signal1,
            signal2,
            fs,
            interval1,
            interval2,
            surrogates,
            window,
            overlap,
            propagation_const,
            order,
            signif,
            *args,
            **kwargs,
        )
    else:
        result = matlab_impl(
            signal1,
            signal2,
            fs,
            interval1,
            interval2,
            surrogates,
            window,
            overlap,
            propagation_const,
            order,
            signif,
            *args,
            **kwargs,
        )

    return result
