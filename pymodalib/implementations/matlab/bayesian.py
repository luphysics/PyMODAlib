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
from pymodalib.utils.matlab import multi_matlab_to_numpy, matlab_to_numpy


@experimental
def bayesian_inference_impl(
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
    *args,
    **kwargs
) -> Any:
    import full_bayesian
    import matlab

    package = full_bayesian.initialize()

    sig1 = matlab.double(signal1.tolist())
    sig2 = matlab.double(signal2.tolist())

    int11, int12 = interval1
    int21, int22 = interval2

    result = package.full_bayesian(
        sig1,
        sig2,
        float(int11),
        float(int12),
        float(int21),
        float(int22),
        float(fs),
        float(window),
        float(propagation_const),
        float(overlap),
        float(order),
        float(surrogates),
        float(signif),
        *args,
        **kwargs,
        nargout=11,
    )

    (
        tm,
        p1,
        p2,
        cpl1,
        cpl2,
        cf1,
        cf2,
        mcf1,
        mcf2,
        surr_cpl1,
        surr_cpl2,
    ) = multi_matlab_to_numpy(*result)

    tm = tm.T

    cpl1 = cpl1.T
    cpl2 = cpl2.T

    surr_cpl1 = surr_cpl1.T
    surr_cpl2 = surr_cpl2.T

    return tm, p1, p2, cpl1, cpl2, cf1, cf2, mcf1, mcf2, surr_cpl1, surr_cpl2
