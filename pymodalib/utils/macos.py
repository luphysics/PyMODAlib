#  PyMODAlib, a Python implementation of the algorithms from MODA (Multiscale Oscillatory Dynamics Analysis).
#  Copyright (C) 2021 Lancaster University
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
import platform
import warnings
from typing import Any, Callable

import multiprocess

from pymodalib.utils.matlab import multi_nested_matlab_to_numpy

_dyld_library_path: str = None
_dyld_key = "DYLD_LIBRARY_PATH"


def should_use_process() -> bool:
    return platform.system() == "Darwin" or "MOCK_MACOS" in os.environ


def configure_dyld_library_path(path: str = None, from_env=False) -> None:
    global _dyld_library_path

    if from_env:
        if _dyld_key not in os.environ:
            warnings.warn(f"Cannot find {_dyld_key} environment variable.")

        _dyld_library_path = os.environ[_dyld_key]
    else:
        _dyld_library_path = path

    if _dyld_key in os.environ:
        del os.environ[_dyld_key]


def run_in_process(func: Callable, *args, **kwargs) -> Any:
    # Not sure why this import is required, but otherwise it's always None.
    from pymodalib.utils.macos import _dyld_library_path as dyld_lib_path

    from scheduler.Scheduler import Scheduler

    scheduler = Scheduler()

    def _func(*args, **kwargs):
        # Not sure why this import is required, but otherwise it's always None.
        from pymodalib.utils.macos import _dyld_library_path as dyld_lib_path

        os.environ[_dyld_key] = dyld_lib_path
        result = func(*args, **kwargs)

        if not isinstance(result, tuple):
            result = (result,)

        return multi_nested_matlab_to_numpy(*result)

    scheduler.add(
        target=_func,
        args=(*args, *kwargs.values()),
        process_type=multiprocess.Process,
        queue_type=multiprocess.Queue,
    )

    return scheduler.run_blocking()[0]
