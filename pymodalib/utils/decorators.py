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
import warnings

from pymodalib.utils import matlab_runtime
from pymodalib.utils.Platform import Platform
from pymodalib.utils.matlab_runtime import (
    MatlabLibraryException,
    platform,
    RuntimeStatus,
)

# Python 3.7 is the highest supported version for MATLAB-packaged libraries.
max_python_version = (3, 7)

msg_no_runtime = (
    f"A compatible MATLAB Runtime version could not be found. "
    + f"Please check the documentation "
    + f"and install the compatible version.\n"
    + f"If you're seeing this error after installing a correct version, please "
    + f"close and re-open "
    + f"your IDE and/or terminals, and if the problem persists, set the Runtime "
    + f"library path:\n{matlab_runtime.get_link_to_runtime_docs()}"
)


def matlabwrapper(module):
    """
    Decorator which marks a MATLAB wrapper: a function which calls a function from a MATLAB-packaged library.

    This will ensure that the required module is installed, and that the MATLAB Runtime is a valid version.

    If the MATLAB Runtime seems to be installed at the default location, but the environment variables are
    not set, it will set the environment variables.

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

            if platform is Platform.WINDOWS:
                runtime_versions = matlab_runtime.get_matlab_runtime_versions()
                if not runtime_versions:
                    raise MatlabLibraryException(
                        f"The MATLAB Runtime is not installed. Please check the documentation "
                        f"for the compatible version and install it."
                    )

                if not matlab_runtime.is_runtime_valid(runtime_versions):
                    raise MatlabLibraryException(msg_no_runtime)
            else:
                status = matlab_runtime.get_runtime_status()

                if status is RuntimeStatus.MAYBE_EXISTS:
                    warnings.warn(
                        "Trying to set MATLAB Runtime variables...", RuntimeWarning
                    )
                    matlab_runtime.try_to_setup_runtime_variables()

                elif status is RuntimeStatus.NOT_EXISTS:
                    raise MatlabLibraryException(msg_no_runtime)

                elif status is not RuntimeStatus.EXISTS:
                    raise Exception("Unknown RuntimeStatus for MATLAB Runtime.")

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


class ExperimentalWarning(UserWarning):
    """
    Warning shown when a function is experimental.
    """


def experimental(func):
    """
    Decorator which marks a function as experimental.
    """

    def wrapper(*args, **kwargs):
        warnings.warn(
            f"The '{func.__name__}' function is experimental. It may produce "
            f"inaccurate results, and its function signature may change in future.",
            ExperimentalWarning,
        )
        return func(*args, **kwargs)

    return wrapper
