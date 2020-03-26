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
Wavelet transform.
"""

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
    *args,
    **kwargs,
) -> Union[Tuple[ndarray, ndarray], Tuple[ndarray, ndarray, Dict]]:
    """
    Wavelet transform function.

    Parameters
    ----------
    signal : ndarray
        [1D array] The signal to perform the wavelet transform on.
    fs : float
        The sampling frequency of the signal.
    fmin: float
         (Default value = None) The minimum frequency for the transform.
    fmax: float
         (Default value = None) The maximum frequency for the transform.
    resolution: float
         (Default value = 1) The frequency resolution for the transform.
    cut_edges: bool
         (Default value = False) Whether to cut the edges of the transform, making the cone of influence 
         visible in the result.
    wavelet : {"Lognorm", "Morlet", "Bump"}, optional
         (Default value = "Lognorm") The type of wavelet transform.
    preprocess: bool
         (Default value = True) Whether to perform pre-processing on the signal before calculating the wavelet transform.
    rel_tolerance: float
         (Default value = 0.01) # TODO docs
    implementation : {"matlab", "python"}, optional
         (Default value = "matlab") Whether to use the MATLAB implementation, or the Python implementation.
         The MATLAB implementation requires the MATLAB Runtime.
    padding: str
         (Default value = "predictive") The type of padding to use when calculating the transform.
    fstep: str
         (Default value = "auto") # TODO docs
    return_opt: bool
         (Default value = False) Whether to return a dictionary containing the options used with the wavelet transform.
    *args : Any, optional
        Any other arguments to pass to the wavelet transform.
    **kwargs : Any, optional
        Any other keyword arguments to pass to the wavelet transform.

    Returns
    -------
    wt : ndarray
        [2D array] The wavelet transform .
    freq : ndarray
        [1D array] The frequencies.
    opt : Dict, optional
        Returned **if and only if** `return_opt==True`. Contains the parameters used by the wavelet transform.
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
            *args,
            **kwargs,
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
            *args,
            **kwargs,
        )
    else:
        raise BadParametersException(
            f"Parameter 'implementation' must be one of: {['matlab', 'python']}."
        )

    if return_opt:
        return wt, freq, opt

    return wt, freq
