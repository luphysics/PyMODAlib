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

import os
import re
import warnings
from enum import Enum
from typing import List, Optional

from pymodalib.utils.Platform import Platform

# The version of the MATLAB Runtime which is required by PyMODAlib.
MATLAB_RUNTIME_VERSION = 96

platform = Platform.get()
regexp = re.compile("v[0-9]{2}")

linux_runtime_var = "LD_LIBRARY_PATH"
linux_runtime_path = (
    f"/usr/local/MATLAB/MATLAB_Runtime/v{MATLAB_RUNTIME_VERSION}/runtime/glnxa64:"
    f"/usr/local/MATLAB/MATLAB_Runtime/v{MATLAB_RUNTIME_VERSION}/bin/glnxa64:"
    f"/usr/local/MATLAB/MATLAB_Runtime/v{MATLAB_RUNTIME_VERSION}/sys/os/glnxa64:"
    f"/usr/local/MATLAB/MATLAB_Runtime/v{MATLAB_RUNTIME_VERSION}/extern/bin/glnxa64"
)

macos_runtime_var = "DYLD_LIBRARY_PATH"
macos_runtime_path = (
    f"/Applications/MATLAB/MATLAB_Runtime/v{MATLAB_RUNTIME_VERSION}/runtime/maci64:"
    f"/Applications/MATLAB/MATLAB_Runtime/v{MATLAB_RUNTIME_VERSION}/bin/maci64:"
    f"/Applications/MATLAB/MATLAB_Runtime/v{MATLAB_RUNTIME_VERSION}/sys/os/maci64:"
    f"/Applications/MATLAB/MATLAB_Runtime/v{MATLAB_RUNTIME_VERSION}/extern/bin/maci64"
)


class RuntimeStatus(Enum):
    """
    Enum representing the status of the MATLAB Runtime.
    """

    EXISTS = 0
    MAYBE_EXISTS = 1
    NOT_EXISTS = 2


def get_runtime_status() -> RuntimeStatus:
    """
    Gets the status of the Matlab Runtime.

    Returns
    -------
    RuntimeStatus
        The status of the MATLAB Runtime.
    """
    if platform is Platform.WINDOWS:
        return RuntimeStatus.MAYBE_EXISTS
    elif platform is Platform.LINUX:
        return _get_runtime_status_linux()
    elif platform is Platform.MAC_OS:
        return _get_runtime_status_macos()


def _split_path(path: str) -> List[str]:
    """
    Splits a path variable, like $PATH, so that each item is separated.

    .. note::
        This is for variables like $PATH, not for file paths.

    Parameters
    ----------
    path : str
        The path to split.

    Returns
    -------
    List[str]
        List containing each item from the path.
    """
    if not path:
        return []

    return path.split(os.pathsep)


def _get_runtime_status_linux() -> RuntimeStatus:
    """
    Returns the RuntimeStatus for Linux.
    """
    spl = _split_path(linux_runtime_path)
    env = os.environ.get(linux_runtime_var)

    if (
        env
        and any([os.path.exists(i) for i in _split_path(env) if i])
        and "runtime" in env.lower()
    ):
        return RuntimeStatus.EXISTS

    if any(os.path.exists(p) for p in spl if p):
        return RuntimeStatus.MAYBE_EXISTS

    return RuntimeStatus.NOT_EXISTS


def _get_runtime_status_macos() -> RuntimeStatus:
    """
    Returns the RuntimeStatus for macOS.
    """
    spl = _split_path(macos_runtime_path)
    env = os.environ.get(macos_runtime_var)

    if (
        env
        and any([os.path.exists(i) for i in _split_path(env) if i])
        and "runtime" in env.lower()
    ):
        return RuntimeStatus.EXISTS

    if any(os.path.exists(p) for p in spl if p):
        return RuntimeStatus.MAYBE_EXISTS

    return RuntimeStatus.NOT_EXISTS


def try_to_setup_runtime_variables() -> None:
    """
    Attempts to setup the MATLAB Runtime environment variables.
    """
    if platform is Platform.LINUX:
        var = linux_runtime_var
        new = linux_runtime_path
    elif platform is Platform.MAC_OS:
        var = macos_runtime_var
        new = macos_runtime_path
    else:
        raise Exception("Cannot try to setup Matlab Runtime on Windows.")

    current = os.environ.get(var)
    os.environ[var] = f"{new}:{current}" if current else new


def is_runtime_valid(versions: List[int]) -> bool:
    """
    Checks whether any of the currently installed MATLAB Runtime versions is valid.

    Parameters
    ----------
    versions : List[int]
        All the currently installed MATLAB Runtime versions, as their version numbers.

    Returns
    -------
    bool
        Whether a compatible MATLAB Runtime is installed.
    """
    return any([v == MATLAB_RUNTIME_VERSION for v in versions])


def get_matlab_runtime_versions() -> List[int]:
    """
    Gets all the installed MATLAB Runtime versions which can be found.

    Returns
    -------
    List[int]
        List containing every MATLAB Runtime version which is installed, as their version numbers (e.g. 96).
    """
    versions = []

    for var in get_path_items(platform):
        if platform is Platform.WINDOWS:
            version = get_runtime_version_windows(var)
        elif platform is not Platform.LINUX and platform is not Platform.MAC_OS:
            raise Exception(
                f"Operating system not recognised. "
                f"Please use Windows, Linux or macOS for running MATLAB-packaged code "
                f"or switch to the pure-Python implementations where possible."
            )

        if version:
            versions.append(version)

    return versions


def get_path_items(platform: Platform) -> List[str]:
    """
    Gets all items from the system PATH.

    Parameters
    ----------
    platform : Platform
        The platform corresponding to OS.

    Returns
    -------
    List[str]
        List containing all the items on the system PATH.
    """
    if platform is Platform.WINDOWS:
        path: str = os.environ.get("path")
    else:
        path: str = os.environ.get("PATH")

    if path:
        return path.split(os.pathsep)

    raise Exception("Environment PATH does not seem to exist.")


def get_runtime_version_windows(var: str) -> Optional[int]:
    """
    Gets the version of the MATLAB Runtime on Windows from a value in an environment variable.

    Parameters
    ----------
    var : str
        The value of the item in the environment variable.

    Returns
    -------
    version: int, None
        If a version can be found, its integer representation is returned (e.g. `96`); otherwise, None.
    """
    if "MATLAB Runtime" in var and "runtime" in var:
        substrings = regexp.findall(var)
        try:
            return int(substrings[0][1:])
        except ValueError or IndexError:
            print(f"Error parsing MATLAB Runtime version from {var}")

    return None


def get_runtime_version_linux(var: str) -> Optional[int]:
    warnings.warn(
        f"Trying to check the MATLAB Runtime version, but this isn't implemented for Linux yet."
    )
    return MATLAB_RUNTIME_VERSION


def get_runtime_version_mac_os(var: str) -> Optional[int]:
    warnings.warn(
        f"Trying to check the MATLAB Runtime version, but this isn't implemented for macOS yet."
    )
    return MATLAB_RUNTIME_VERSION


class MatlabLibraryException(Exception):
    """
    Exception raised when a MATLAB-packaged library is missing.
    """


def get_link_to_runtime_docs() -> str:
    """
    Gets a link to the section of the MATLAB documentation which explains how to add the MATLAB Runtime to the PATH.

    Returns
    -------
    str
        The URL for the relevant page of the MATLAB documentation.
    """
    return {
        Platform.WINDOWS.value: "https://uk.mathworks.com/help/matlab/matlab_external/building-and-running-eng"
        "ine-applications-on-windows-operating-systems.html",
        Platform.LINUX.value: "https://uk.mathworks.com/help/matlab/matlab_external/set-run-time-library-path"
        "-on-linux-systems.html",
        Platform.MAC_OS.value: "https://uk.mathworks.com/help/matlab/matlab_external/set-run-time-library-path"
        "-on-mac-systems.html",
    }.get(platform.value)
