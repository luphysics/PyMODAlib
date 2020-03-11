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

# The version of the MATLAB Runtime required.
import os
import re
import warnings
from typing import List, Optional

from pymodalib.utils.Platform import Platform

# The version of the MATLAB Runtime which is required by PyMODAlib.
MATLAB_RUNTIME_VERSION = 96

platform = Platform.get()
regexp = re.compile("v[0-9]{2}")


def is_runtime_valid() -> bool:
    warnings.warn(
        f"Skipping a check for the validity of the MATLAB Runtime, "
        f"because this functionality is not fully implemented yet.",
        RuntimeWarning,
    )
    # versions = get_matlab_runtime_versions()
    # return any([v == MATLAB_RUNTIME_VERSION for v in versions])
    return True  # TODO: fix this function


def get_matlab_runtime_versions() -> List[int]:
    versions = []

    for var in get_path_items(platform):
        if platform is Platform.WINDOWS:
            version = get_runtime_version_windows(var)
        elif platform is Platform.LINUX:
            version = get_runtime_version_linux(var)
        elif platform is Platform.MAC_OS:
            version = get_runtime_version_mac_os(var)
        else:
            raise Exception(
                f"Operating system not recognised. "
                f"Please use Windows, Linux or macOS for running MATLAB-packaged code "
                f"or switch to the pure-Python implementations where possible."
            )

        versions.append(version)

    return versions


def get_path_items(platform: Platform) -> List[str]:
    if platform is Platform.WINDOWS:
        path: str = os.environ.get("path")
    else:
        path: str = os.environ.get("PATH")

    if path:
        return path.split(os.pathsep)

    raise Exception("Environment PATH does not seem to exist.")


def get_runtime_version_windows(var: str) -> Optional[int]:
    if "MATLAB Runtime" in var and "runtime" in var:
        substrings = regexp.findall(var)
        try:
            return int(substrings[0][1:])
        except ValueError or IndexError:
            print(f"Error parsing MATLAB Runtime version from {var}")

    return None


def get_runtime_version_linux(var: str) -> Optional[int]:
    raise NotImplementedError("Linux not implemented yet.")


def get_runtime_version_mac_os(var: str) -> Optional[int]:
    raise NotImplementedError("macOS not implemented yet.")


def raise_invalid_exception() -> None:
    raise MatlabRuntimeException(
        f"MATLAB Runtime is not installed or compatible. "
        f"Please install version v{MATLAB_RUNTIME_VERSION}."
    )


class MatlabRuntimeException(Exception):
    pass
