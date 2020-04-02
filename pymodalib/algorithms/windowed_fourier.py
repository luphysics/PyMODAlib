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
Windowed Fourier transform.
"""

from typing import Tuple

from numpy import ndarray


def windowed_fourier_transform(
    signal: ndarray,
    fs: float,
    fmin: float,
    fmax: float = None,
    resolution: float = None,
    cut_edges: bool = False,
    window: str = "Gaussian",
    preprocess: bool = True,
    fstep: str = "auto",
    padding: str = "predictive",
    rel_tolerance: float = 0.01,
    implementation: str = "matlab",
    *args,
    **kwargs,
) -> Tuple[ndarray, ndarray]:
    """
    Windowed Fourier transform function.

    .. note::
        Although the wavelet transform function and the WFT MATLAB library can return a dictionary containing
        parameters used by the function, `windowed_fourier_transform` does not support this.

    Parameters
    ----------
    signal : ndarray
       [1D array] The signal to perform the windowed Fourier transform on.
    fs : float
       The sampling frequency of the signal.
    fmin : float
       The minimal frequency for which to calculate the WFT.
    fmax : float
       (Default = `fs/2`) The maximal frequency for which to calculate the WFT.
    resolution : float
       (Default = `1/fmin`) The window resolution parameter, which determines the trade-off between the time
       and frequency resolutions; the higher it is, the closer in frequency components can be resolved in WFT,
       but the closer the slower time-variations, e.g. amplitude/frequency modulation, can be reliably represented.
       For the way it is introduced for each window, see Appendix E in [1], while if the window is user-defined in
       terms of its function in frequency and/or time (see `window` parameter), then `resolution` has no effect.
    cut_edges : bool
        (Default = False) Whether WFT coefficients should be set to NaN out of the influence (see [1]).
        Use `cut_edges=True` if you wish to analyze the WFT only within the cone of influence, which is recommended
        if you are estimating only the time-averaged quantities.
    window : {"Gaussian", "Hann", "Blackman", "Exp", "Rect", "Kaiser-a"}
        (Default = "Gaussian") Window used in the WFT calculation.
        For a list of all supported names and their properties, see Appendix E in [1].
        *Note: supplying a window using a custom function is not supported in PyMODAlib.*
    preprocess : bool
        (Default = True) Whether to perform signal preprocessing, which consists of subtracting third-order
        polynomial fit and then bandpassing the signal in the band of interest (`fmin`-`fmax`).
    fstep : {"auto", "auto-NB"}
        (Default = "auto", equivalent to "auto-10") The frequency step, which determines frequency discretization,
        so that the next frequency
        equals the previous frequency plus `fstep`. When set to `auto-NB`, e.g. `auto-20`, it determines automatically
        as described in [1], so that it equals `1/NB` of the frequency region containing 50% of the window function.
    padding : {"predictive", 0, "symmetric", "none", "periodic"}, float
        (Default = "predictive") Padding to use when calculating the transform. For all paddings and their effects,
        see [1].
        Most useful are the zero-padding, for which boundary errors are well-determined, and "predictive" padding,
        for which they are most reduced, while other choices have limited usefulness.
        *Note: the if a List containing two padding parameters from the accepted values is passed, the first value
        will be used for left-padding and the second value for right-padding.*
    rel_tolerance : float
        (Default = 0.01) Commonly referred to as `epsilon` in [1], this parameter is the relative tolerance as a
        percentage, which specifies the cone of influence for the WFT (i.e. the range of WFT coefficients which
        are determined up to this accuracy in respect of boundary errors). Also determintes the minimal number of values
        to pad the signal with,so that the relative constribution of effects of implicit periodic signal continutation
        due to convolution in the frequency domain is smaller. See [1] for details.
    implementation : {"matlab"}
        (Default = "matlab") Whether to use the MATLAB implementation, or (possibly, in the future) the
        Python implementation.
    args
        Other arguments to pass to the implementation (the MATLAB-packaged WFT function).
    kwargs
        Other keyword arguments to pass to the implementation (the MATLAB-packaged WFT function).

    Returns
    -------
    wft : ndarray
        [2D array, complex] The windowed Fourier transform, whose rows correspond to frequencies and columns
        to time. Take the absolute value to get the amplitude.
        **Dimensions: (FNxL) where FN is the number of frequencies and L is the length of the signal in samples.**
    freq : ndarray
        [1D array] The frequencies corresponding to the rows of `wft`.

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
    if not fmax:
        fmax = fs / 2

    if not resolution:
        resolution = 1 / fmin

    if implementation != "matlab":
        raise NotImplementedError(
            f"No Python implementation for the windowed Fourier transform."
        )
    else:
        from pymodalib.implementations.matlab.windowed_fourier import wft_impl

        wft, freq = wft_impl(
            signal=signal,
            fs=fs,
            fmin=fmin,
            fmax=fmax,
            resolution=resolution,
            cut_edges=cut_edges,
            window=window,
            preprocess=preprocess,
            padding=padding,
            fstep=fstep,
            rel_tolerance=rel_tolerance,
            *args,
            **kwargs,
        )

    return wft, freq
