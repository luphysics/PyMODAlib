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

# Import functions to make them easily accessible.
from pymodalib.algorithms.group_coherence import group_coherence, dual_group_coherence
from pymodalib.algorithms.harmonics import harmonicfinder
from pymodalib.algorithms.ridge_extraction import ridge_extraction, ecurve, rectfr
from pymodalib.algorithms.signals import preprocess, generate_times
from pymodalib.algorithms.signals import resampl_flow as downsample
from pymodalib.algorithms.wavelet import wavelet_transform
from pymodalib.algorithms.windowed_fourier import windowed_fourier_transform
from pymodalib.utils.plotting import colormap, contourf
from pymodalib.utils.cache import cachedarray, cleanup

# PyMODAlib version.
__version__ = "0.8.1b1"

# This tuple isn't important; it just ensures that PyCharm doesn't try to remove unnecessary imports.
__imported = (
    cachedarray,
    cleanup,
    group_coherence,
    dual_group_coherence,
    downsample,
    preprocess,
    generate_times,
    wavelet_transform,
    windowed_fourier_transform,
    harmonicfinder,
    ridge_extraction,
    ecurve,
    rectfr,
    colormap,
    contourf,
)

# Useful aliases for functions.
wt = wavelet_transform
wft = windowed_fourier_transform
