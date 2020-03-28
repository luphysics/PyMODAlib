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
import sys

from pymodalib.utils import matlab_runtime
from pymodalib.utils.matlab import MatlabLibraryException

# Python 3.7 is the highest supported version for MATLAB-packaged libraries.
max_python_version = (
    3,
    7,
)


def matlabwrapper(module):
    """
    Decorator which marks a MATLAB wrapper: a function which calls a function from a MATLAB-packaged library.

    This will ensure that the required module is installed, and that the MATLAB Runtime is a valid version.

    Parameters
    ----------
    module : str
        The name of the MATLAB-packaged library which is being used.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            import importlib.util

            if sys.version_info[:2] > max_python_version:
                raise MatlabLibraryException(
                    f"MATLAB-packaged libraries are only supported on Python "
                    f"{'.'.join([str(s) for s in max_python_version])} and below. "
                    f"Please try using a compatible "
                    f"version of Python, or check if there is a pure-Python implementation "
                    f"by passing 'implementation=\"python\"' to the function."
                )

            module_valid = importlib.util.find_spec(module)
            if not module_valid:
                raise MatlabLibraryException(
                    f"The MATLAB-packaged library '{module}' is not installed. Please install "
                    f"PyMODA, which supplies the MATLAB-packaged libraries, "
                    f"or check if there is a "
                    f"pure-Python implementation by passing 'implementation=\"python\"' to "
                    f"the function."
                )

            runtime_valid = matlab_runtime.is_runtime_valid()

            if not runtime_valid:
                matlab_runtime.raise_invalid_exception()

            return func(*args, **kwargs)

        return wrapper

    return decorator


def deprecated(func):
    """
    Decorator that marks a function as deprecated, i.e. that it should no
    longer be used and will be removed in future.
    """

    def wrapper(*args, **kwargs):
        print(f"Warning: calling deprecated function '{func.__name__}'.")
        return func(*args, **kwargs)

    return wrapper
