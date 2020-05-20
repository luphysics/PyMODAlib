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

from typing import Tuple

from numpy import ndarray
from pymodalib.implementations.python.wavelet.wavelet_transform import MorseWavelet

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
    implementation="python",
    padding: str = "predictive",
    return_opt: bool = False,
    parallel: bool = None,
    *args,
    **kwargs,
) -> Tuple[ndarray, ndarray]:
    """
    Wavelet transform function.

    .. note::
        If `return_opt == True`, a dictionary will be returned in addition to the normal return values.
        This dictionary contains parameters used by the function.

    Parameters
    ----------
    signal : ndarray
       [1D array] The signal to perform the wavelet transform on.
    fs : float
       The sampling frequency of the signal.
    fmin : float
        The minimal frequency for which to calculate the WT. If left to the default value, the function will use the
        minimal frequency for which at least one WT coefficient is determined up to a specified relative
        accuracy (`rel_tolerance`) with respect to boundary errors.
    fmax : float
       (Default = `fs/2`) The maximal frequency for which to calculate the WT.
    resolution : float
       (Default = 1) The wavelet resolution parameter, which determines the trade-off between the time
       and frequency resolutions; the higher it is, the closer in frequency components can be resolved in WT,
       but the closer the slower time-variations, e.g. amplitude/frequency modulation, can be reliably represented.
       For the way it is introduced for each wavelet, see Appendix E in [1], while if the wavelet is user-defined in
       terms of its function in frequency and/or time (see `wavelet` parameter), then `resolution` has no effect.
    cut_edges : bool
        (Default = False) Whether WT coefficients should be set to NaN out of the influence (see [1]).
        Use `cut_edges=True` if you wish to analyze the WT only within the cone of influence, which is recommended
        if you are estimating only the time-averaged quantities.
    wavelet : {"Lognorm", "Morlet", "Morse-a"}
        (Default = "Lognorm") Wavelet used in the WT calculation.
        For a list of all supported names and their properties, see Appendix E in [1].
        *Note: supplying a wavelet using a custom function is not supported in PyMODAlib.*
    preprocess : bool
        (Default = True) Whether to perform signal preprocessing, which consists of subtracting third-order
        polynomial fit and then bandpassing the signal in the band of interest (`fmin`-`fmax`).
    padding : {"predictive", 0, "symmetric", "none", "periodic"}, float
        (Default = "predictive") Padding to use when calculating the transform. For all paddings and their effects,
        see [1].
        Most useful are the zero-padding, for which boundary errors are well-determined, and "predictive" padding,
        for which they are most reduced, while other choices have limited usefulness.
        *Note: if a List containing two padding parameters from the accepted values is passed, the first value
        will be used for left-padding and the second value for right-padding.*
    rel_tolerance : float
        (Default = 0.01) Commonly referred to as `epsilon` in [1], this parameter is the relative tolerance as a
        percentage, which specifies the cone of influence for the WT (i.e. the range of WT coefficients which
        are determined up to this accuracy in respect of boundary errors). Also determintes the minimal number of values
        to pad the signal with,so that the relative constribution of effects of implicit periodic signal continutation
        due to convolution in the frequency domain is smaller. See [1] for details.
    implementation : {"matlab", "python"}, optional
        (Default value = "python") Whether to use the MATLAB implementation, or the Python implementation.
        The MATLAB implementation requires the MATLAB Runtime.
    return_opt : bool
         (Default value = False) Whether to return a dictionary containing the parameters used with the wavelet
         transform. This can be useful if `fmin` was left to its default value, since it will contain the value
         of `fmin` which was used.
    parallel : bool, None
        Whether to parallelize the algorithm. This may improve performance by up to 25% for larger signals,
        but it also increases memory usage.
    *args : Any, optional
        Any other arguments to pass to the wavelet transform implementation.
    **kwargs : Any, optional
        Any other keyword arguments to pass to the wavelet transform implementation.

    Returns
    -------
    wt : ndarray
        [2D array, complex] The wavelet transform, whose rows correspond to frequencies and columns
        to time. Take the absolute value to get the amplitude.
        **Dimensions: (FNxL) where FN is the number of frequencies and L is the length of the signal in samples.**
    freq : ndarray
        [1D array] The frequencies corresponding to the rows of `wt`.

    Notes
    -----
    Author: Dmytro Iatsenko.

    .. [1] D. Iatsenko, A. Stefanovska and P.V.E. McClintock,
       "Linear and synchrosqueezed time-frequency representations revisited.
       Part I: Overview, standards of use, related issues and algorithms."
       {preprint:arXiv:1310.7215}
    .. [2] D. Iatsenko, A. Stefanovska and P.V.E. McClintock,
       "Linear and synchrosqueezed time-frequency representations revisited.
       Part II: Resolution, reconstruction and concentration."
       {preprint:arXiv:1310.7274}
    """
    verify_parameter(wavelet, possible_values=["Lognorm", "Bump", "Morlet", "Morse-a"])
    verify_parameter(implementation, possible_values=["matlab", "python"])

    if implementation == "python":
        from pymodalib.implementations.python.wavelet.wavelet_transform import (
            LognormWavelet,
            MorletWavelet,
            wavelet_transform as python_impl,
        )

        if wavelet == "Lognorm":
            wp = LognormWavelet(resolution)
        elif wavelet == "Morlet":
            wp = MorletWavelet(resolution)
        elif wavelet == "Bump":
            raise Exception("Bump wavelet is not supported yet.")
        elif wavelet == "Morse-a":
            wp = MorseWavelet(3, resolution)
        else:
            raise ValueError(f"Unknown wavelet: '{wavelet}'")

        result = python_impl(
            signal=signal,
            fs=fs,
            wp=wp,
            fmin=fmin,
            fmax=fmax,
            cut_edges=cut_edges,
            preprocess=preprocess,
            padding=padding,
            rel_tolerance=rel_tolerance,
            return_opt=return_opt,
            parallel=parallel,
            *args,
            **kwargs,
        )

        if return_opt:
            wt, freq, opt = result
        else:
            wt, freq = result

    elif implementation == "matlab":
        from pymodalib.implementations.matlab.wavelet.wavelet_transform import (
            wavelet_transform as matlab_impl,
        )

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
