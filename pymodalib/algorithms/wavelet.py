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
from typing import Union, Tuple, Dict

from numpy import ndarray

from pymodalib.implementations.matlab.wavelet.wavelet_transform import (
    wavelet_transform as matlab_impl,
)
from pymodalib.implementations.python.wavelet.wavelet_transform import (
    wavelet_transform as python_impl,
)
from pymodalib.utils.parameters import verify_parameter, BadParametersException


def wavelet_transform(
    signal: ndarray,
    fs: float,
    fmin: float = None,
    fmax: float = None,
    resolution: float = 1,
    cut_edges: bool = False,
    wavelet: str = "Lognorm",
    preprocess: bool = True,
    rel_tolerance: float = 0.01,
    implementation="matlab",
    padding: str = "predictive",
    fstep: str = "auto",
    return_opt: bool = False,
) -> Union[Tuple[ndarray, ndarray], Tuple[ndarray, ndarray, Dict]]:
    """
    Wavelet transform function.

    :param signal: the signal to perform the wavelet transform on
    :param fs: the sampling frequency of the signal
    :param fmin: the minimum frequency for the transform
    :param fmax: the maximum frequency for the transform
    :param resolution:
    :param cut_edges:
    :param wavelet:
    :param preprocess:
    :param rel_tolerance:
    :param implementation:
    :param fstep:
    :param padding:
    :param return_opt:
    :return: [2D array] the wavelet transform; [1D array] the frequencies
    """
    verify_parameter(wavelet, possible_values=["Lognorm", "Bump", "Morlet"])
    verify_parameter(implementation, possible_values=["matlab", "python"])

    if implementation == "python":
        wt, freq, opt = python_impl(
            signal,
            fs,
            fmin,
            fmax,
            resolution,
            cut_edges,
            wavelet,
            preprocess,
            rel_tolerance,
        )
    elif implementation == "matlab":
        wt, freq, opt = matlab_impl(
            signal=signal,
            fs=fs,
            fmin=fmin,
            fmax=fmax,
            resolution=resolution,
            cut_edges=cut_edges,
            wavelet=wavelet,
            preprocess=preprocess,
            padding=padding,
            fstep=fstep,
            rel_tolerance=rel_tolerance,
        )
    else:
        raise BadParametersException(
            f"Parameter 'implementation' must be one of: {['matlab', 'python']}."
        )

    if return_opt:
        return wt, freq, opt

    return wt, freq
