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
import warnings
from typing import Tuple, Dict, Union

from numpy import ndarray

from pymodalib.implementations.matlab.ridge_extraction import ecurve_impl, rectfr_impl
from pymodalib.utils import reorient
from pymodalib.utils.matlab import matlab_to_numpy, multi_matlab_to_numpy


def ridge_extraction(
    tfr: ndarray,
    frequencies: ndarray,
    fs: float,
    method: str = "direct",
    wopt: Dict = None,
    ecurve_kwargs: Dict = None,
    **kwargs,
) -> Tuple[ndarray, ndarray, ndarray]:
    """
    Ridge extraction algorithm.

    Parameters
    ----------
    tfr : ndarray
        Time-frequency representation of the signal, i.e. WT or WFT.
    frequencies : ndarray
        Frequencies corresponding to the `tfr`.
    fs : float
        Sampling frequency of the signal.
    method : {"direct", "ridge", "both"}
        (Default = "direct") The reconstruction method to use for estimating the component's
        parameters (`iamp`, `iphi`, `ifreq` - see [1]). If set to "both", all parameters are returned
        as [2xL arraus] with direct and ridge estimates corresponding to the 1st and 2nd rows, respectively.
    wopt : Dict
        Dictionary returned by the wavelet transform function when the `return_opt=True` parameter is passed.
        This should be used when possible.
    ecurve_kwargs : Any
        Dictionary containing keyword arguments to pass to the `ecurve` function.
    kwargs : Any
        Additional keyword arguments.

    Returns
    -------
    iamp: ndarray
        Component's amplitude, as reconstructed from `tfsupp` in the signal's time-frequency representation.
    iphi : ndarray
        Component's phase, as reconstructed... (see above)
    ifreq : ndarray
        Component's frequency, as reconstructed... (see above)

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
    .. [3] D. Iatsenko, A. Stefanovska and P.V.E. McClintock,
       "On the extraction of instantaneous frequencies from ridges in
       time-frequency representations of signals."
       {preprint - arXiv:1310.7276}
    """
    if not wopt:
        warnings.warn(
            f"'wopt' was not passed to the ridge extraction function. It's recommended that you "
            f"get the value of 'wopt' by passing 'return_opt=True' to the wavelet transform,"
            f"and pass it to the ridge extraction function."
        )

    wopt = wopt or {}
    ecurve_kwargs = ecurve_kwargs or {}

    if wopt.get("TFRname") == "WFT":
        warnings.warn(
            "ridge_extraction may not be compatible with WFT in PyMODAlib.",
            RuntimeWarning,
        )

    # Calculate time-frequency support.
    tfsupp = ecurve(
        tfr,
        frequencies,
        fs=fs,
        wopt=wopt,
        method=method,
        return_matlab_types=True,  # Use MATLAB data type to remove overhead due to conversion.
        **ecurve_kwargs,
        **kwargs,
    )

    iamp, iphi, ifreq = rectfr(
        tfsupp, tfr, frequencies, fs=fs, wopt=wopt, method=method,
    )

    iamp = reorient(iamp)
    iphi = reorient(iphi)
    ifreq = reorient(ifreq)

    return iamp, iphi, ifreq


def ecurve(
    tfr: ndarray,
    frequencies: ndarray,
    fs: float,
    method: Union[int, str, ndarray] = None,
    param: Union[float, Tuple[float, float]] = None,
    normalize: bool = None,
    wopt: Dict = None,
    path_optimize: bool = None,
    max_iterations: int = None,
    return_matlab_types=False,
    **kwargs,
) -> Tuple[ndarray]:
    """
    Extracts the curve (i.e. the sequence of the amplitude ridge points) and its
    full support (the widest region of unimodal TFR amplitude around them) from the given
    time-frequency representation of the signal. The TFR can be either the wavelet transform
    or the windowed Fourier transform.

    Parameters
    ----------
    tfr : ndarray
        [NFxL array] Time-frequency representation (wavelet transform or windowed Fourier transform), to extract from.
    frequencies : ndarray
        [NFx1 array] The frequencies corresponding to the rows of `tfr`.
    fs : float
        Sampling frequency of the signal.
    method : {"direct", "ridge", "both"}
        (Default = "direct") The reconstruction method to use for estimating the component's
        parameters (`iamp`, `iphi`, `ifreq`) (see [1]); if set to "both", all parameters are returned as [2xL arrays]
        with direct and ridge estimates corresponding to the 1st and 2nd rows respectively.
    param : [float, float], float, optional
        Parameters for each method, as described in [3].
    normalize : bool
        (Default = False) Noise power in time-frequency domain can depend on frequency, so
        that at lowest or highest frequencies the noise-induced amplitude
        peaks might overgrow the peaks associated with a genuine components,
        thus being selected instead of the latter. To avoid this, the curve
        can be extracted using the normalized amplitude peaks, that are
        non-uniformly reduced in dependence on their frequencies, with a
        suitable normalization being determined based on the dependence of
        the mean TFR amplitude on frequency; setting `normalize` to True
        applies such a normalization.
    wopt : Dict
        Dictionary returned by the wavelet transform function when the `return_opt=True` parameter is passed.
        This should be used when possible.
    path_optimize : bool
        (Default = True) Optimize the ridge curve over all possible trajectories (True) or
        use the one-step approach (False), see [3]; the path optimization
        GREATLY improves the performance of all methods and is not
        computationally expensive (is performed in O(N) operations using
        fast algorithm of [3]), so DO NOT CHANGE THIS PROPERTY unless you
        want to just play and see the advantages of the path optimization
        over the one-step approach; note, that this property applies only
        to methods 1 and 2, as the others are simple one-step approaches.
    max_iterations : int
        (Default = 20) Maximum number of iterations allowed for methods 2 and 3 to converge.
    return_matlab_types : bool
        (Default = False) Whether to return the results as MATLAB types, instead of converting.
        This can reduce overhead when calling another MATLAB-packaged function with the result,
        because the data doesn't need to be converted from MATLAB to Numpy and then back to MATLAB.
    kwargs

    Returns
    -------
    tfsupp : ndarray
        [3xL array] The extracted time-frequency support of the component, containing frequencies of the
        TFR amplitude peals (ridge points) in the first row, support lower bounds (referred to as `omega_-(t)/2/pi`
        in [1]) in the second row, and the upper bounds (referred to as `omega_+(t)/2/pi` in [1]) in the third row.

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
    .. [3] D. Iatsenko, A. Stefanovska and P.V.E. McClintock,
       "On the extraction of instantaneous frequencies from ridges in
       time-frequency representations of signals."
       {preprint - arXiv:1310.7276}
    """
    tfsupp = ecurve_impl(
        tfr,
        frequencies,
        fs,
        method=method,
        param=param,
        normalize=normalize,
        wopt=wopt,
        path_optimize=path_optimize,
        max_iterations=max_iterations,
        **kwargs,
    )

    if not return_matlab_types:
        tfsupp = matlab_to_numpy(tfsupp)

    return tfsupp


def rectfr(
    tfsupp: ndarray,
    tfr: ndarray,
    frequencies: ndarray,
    fs: float,
    wopt: Dict = None,
    method: str = "direct",
) -> Tuple[ndarray, ndarray, ndarray]:
    """
    Calculates the component's amplitude, phase and frequency as reconstructed from its
    extracted time-frequency support in the signal's transform (WT/WFT).

    Parameters
    ----------
    tfsupp : ndarray, matlab.double
        [3xL array] The extracted time-frequency support of the component, containing frequencies of the
        TFR amplitude peals (ridge points) in the first row, support lower bounds (referred to as `omega_-(t)/2/pi`
        in [1]) in the second row, and the upper bounds (referred to as `omega_+(t)/2/pi` in [1]) in the third row.

        Alternatively, `tfsupp` can be supplied as a [1xL array] of the desired frequency profile; in this case, the
        algorithm will automatically select the time-frequency support around it and the corresponding peaks.
    tfr : ndarray, matlab.double
        [NFxL array] The transform (wavelet transform or windowed Fourier transform), to which `tfsupp` corresponds.
        (Rows correspond to frequencies, columns to time.)
    frequencies : ndarray, matlab.double
        [NFx1 array] The frequencies corresponding to the rows of `transform`.
    fs : float
        Sampling frequency of the signal.
    wopt : Dict
        Dictionary returned by the wavelet transform function when the `return_opt=True` parameter is passed.
        This should be used when possible.
    method : {"direct", "ridge", "both"}
        (Default = "direct") The reconstruction method to use for estimating the component's
        parameters (`iamp`, `iphi`, `ifreq`) (see [1]); if set to "both", all parameters are returned as [2xL arrays]
        with direct and ridge estimates corresponding to the 1st and 2nd rows respectively.

    .. note::
        In the case of direct reconstruction, if the window/wavelet does not allow direct estimation of frequency,
        then the frequency is reconstructed by the hybrid method (see [1]).

    Returns
    -------
    iamp: ndarray
        Component's amplitude, as reconstructed from `tfsupp` in the signal's time-frequency representation.
    iphi : ndarray
        Component's phase, as reconstructed... (see above)
    ifreq : ndarray
        Component's frequency, as reconstructed... (see above)

    Notes
    -----
    Author: Dmytro Iatsenko

    .. [1] D. Iatsenko, A. Stefanovska and P.V.E. McClintock,
       "Linear and synchrosqueezed time-frequency representations revisited.
       Part I: Overview, standards of use, related issues and algorithms."
       {preprint:arXiv:1310.7215}
    .. [2] D. Iatsenko, A. Stefanovska and P.V.E. McClintock,
       "Linear and synchrosqueezed time-frequency representations revisited.
       Part II: Resolution, reconstruction and concentration."
       {preprint:arXiv:1310.7274}
    """

    iamp, iphi, ifreq = rectfr_impl(
        tfsupp, tfr, frequencies, fs=fs, method=method, wopt=wopt,
    )

    return multi_matlab_to_numpy(iamp, iphi, ifreq)
