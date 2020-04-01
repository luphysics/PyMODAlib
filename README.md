# PyMODAlib

[![DOI](https://zenodo.org/badge/243930888.svg)](https://zenodo.org/badge/latestdoi/243930888)
[![License: GPL](https://img.shields.io/badge/License-GPLv3-10b515.svg)](https://github.com/luphysics/PyMODAlib/blob/master/LICENSE)
[![PyPI: version](https://img.shields.io/pypi/v/PyMODAlib)](https://pypi.org/project/PyMODAlib)
[![PyPI: Python version](https://img.shields.io/pypi/pyversions/PyMODAlib)](https://pypi.org/project/PyMODAlib)
[![Documentation Status](https://readthedocs.org/projects/pymodalib/badge/?version=latest)](https://pymodalib.readthedocs.io/en/latest/?badge=latest)
[![Code style: Black](https://img.shields.io/badge/code_style-black-black)](https://github.com/psf/black)

## Introduction

PyMODAlib is a Python library containing the algorithms used by [PyMODA](https://github.com/luphysics/PyMODA). With PyMODAlib, you can write Python scripts to perform the same calculations as PyMODA.

Some of PyMODAlib's algorithms are MATLAB-packaged libraries, while some are Python translations of algorithms belonging to [MODA](https://github.com/luphysics/MODA).

### License

You may use, distribute and modify this software under the terms of the [GNU General Public License v3.0](https://opensource.org/licenses/GPL-3.0). See [LICENSE](/LICENSE).

### References and citations

To cite PyMODAlib or view its references, please see the DOI at [Zenodo](https://zenodo.org/badge/latestdoi/243930888).

## User Guide

This section describes how to get started with PyMODAlib. 

For a full API reference, please see PyMODAlib's [ReadTheDocs](https://pymodalib.readthedocs.io/) page, which shows the parameters and output for every function.

### Prerequisites

PyMODAlib requires Python 3.6 or higher. Some features also require the MATLAB Runtime, version 9.6.

> **Note:** See [current status](#current-status) to check which functions require the MATLAB Runtime.

### Installing PyMODAlib

PyMODAlib can be installed using `pip`. Open a terminal and run:

```bash
pip install pymodalib
```

> **Tip:** on macOS/Linux, replace `pip` with the correct command for your system (e.g. `pip3`).

### Updating PyMODAlib

PyMODAlib will be updated regularly. To update your installed version, open a terminal and run:

```bash
pip install -U pymodalib
```

### Current status

`PyMODAlib` is still in development. Currently, the features implemented are:

| Feature | Implemented | Requires MATLAB Runtime | Notes |
| --- | --- | --- | --- |
| Wavelet transform | :heavy_check_mark: | No | Pass `implementation="python"` to use Python implementation | 
| Wavelet phase coherence | :heavy_check_mark: | No | | 
| Group coherence | :heavy_check_mark: | No | Pass `implementation="python"` to use Python implementation | 
| Detecting harmonics | :heavy_check_mark: | No | |
| Downsampling | :heavy_check_mark: | No | | 

> :warning: Some functions may not be fully stable, and may change before `v1.0.0`.

### Getting started

#### Examples

There are examples of using PyMODAlib's functionality in the [examples](https://github.com/luphysics/PyMODAlib/tree/master/examples) directory.

To download the dependencies required to run the examples, open a terminal and run:

```bash
pip install -U pymodalib matplotlib
```

To try the examples, download the PyMODAlib repository [as a zip file](https://github.com/luphysics/PyMODAlib/zipball/master) or by using `git clone`, then run Python files from the `examples` subfolders.

### PyMODAlib cache

The group coherence functions use a very large quantity of RAM. To mitigate this problem for machines with smaller RAM capacities, they will allocate arrays which are cached to disk. *This may result in significant disk usage.*

By default, PyMODAlib will use a folder named `.pymodalib` inside your home directory for its cache. However, it will show a `RuntimeWarning` unless you set the location manually. This warning is intended to make users aware of the risks of placing the cache folder on an SSD.

> :warning: If the cache folder is on an SSD, it may **reduce the lifespan** of the SSD.

#### Setting the cache location

To set the location of the cache folder manually, use the `PYMODALIB_CACHE` environment variable. Instructions for different operating systems are below.

***The cache location should be set to an empty folder which resides on an HDD.***

##### Windows

- Create a folder to use for the cache. 
- Press the start button and type "environment" until the option "Edit the system environment variables" appears, and click it. 
- Click "Environment variables" near the bottom right of the dialog.
- Click "New" in the "System variables" section of the window which appears. 
- In the dialog which opens, enter "PYMODALIB_CACHE" as the variable name and click "Browse Directory" to choose the folder you created.
- Press "Ok" to close all dialogs.
- Restart your IDE and/or terminal.

##### Linux

Create a folder to use for the cache. Open a terminal and `cd` to the folder, then copy the following commands into the terminal:

```bash
echo "export PYMODALIB_CACHE=$(pwd)" >> ~/.bashrc
source ~/.bashrc
```

You may need to restart your IDE and any other open terminals.

##### macOS 

Create a folder to use for the cache. Open a terminal and `cd` to the folder, then copy the following commands into the terminal:

```bash
echo "export PYMODALIB_CACHE=$(pwd)" >> ~/.bash_profile
source ~/.bash_profile
```

You may need to restart your IDE and any other open terminals.

## Developer guide

This guide is aimed at developers interested in contributing to PyMODAlib. 

### Downloading the code

To download the code, you should fork the repository and clone your fork.

### Installing the requirements

Open a terminal in the `PyMODAlib` folder and run:

```bash
pip install -r requirements.txt
pip install pre-commit
```

### Git hooks

Git hooks are used to automatically format modified Python files with [Black](https://github.com/psf/black) when a commit is made. This ensures that the code follows a uniform style.

To install the Git hooks, open a terminal in the `PyMODAlib` folder and run:

```bash
pre-commit install
```

> **Tip:** If this causes an error, try `python -m pre-commit install`.

When you make a commit, the modified Python files will be formatted if necessary. If this occurs, you'll need to repeat your `git add` and `git commit` commands to make the commit. 

> **Tip:** You can still use PyCharm's auto-formatter while writing code. Although it sometimes disagrees with `black`, `black` will undo its changes at commit-time, so no harm is done.

### Developing PyMODAlib

When developing PyMODAlib, you can test your changes by installing the library locally in "editable" mode. From the root of the repository, run:

```bash
pip install -e .
```

> **Note:** After making changes to PyMODAlib, you don't need to run the `pip install` command again. Any Python script which imports `pymodalib` will reflect the changes immediately.

Switching back to the release version of PyMODAlib is simple:

```bash
pip uninstall pymodalib -y
pip install -U pymodalib
```

### Project structure

The public-facing API is located in the `algorithms` package. This package contains wrappers for the actual implementations, which can be found in the `implementations` package.

This structure allows the implementation to be easily changed, while ensuring that the API remains backwards-compatible.

#### `__init__.py`

In `pymodalib.__init__.py`, many functions are imported from the `algorithms` package. This allows users to more easily find useful functions: for example, they can use `pymodalib.wavelet_transform` instead of `pymodalib.algorithms.wavelet.wavelet_transform`.

#### `implementations` package

The `implementations` package contains a `matlab` package and a `python` package. The `matlab` package contains wrappers for algorithms supplied by MATLAB-packaged libraries, while the `python` package contains algorithms implemented purely in Python.

### Docstring style

PyMODAlib uses [Numpy-style docstrings](https://numpydoc.readthedocs.io/en/latest/format.html). 

To configure PyCharm to use this style, go to `Settings` -> `Tools` -> `Python Integrated Tools` and change `Docstring format` from `reStructuredText` to `NumPy`.

### "Gotchas"

When importing a function/package from another part of `pymodalib`, your IDE may auto-import the package without the `pymodalib` prefix. For example, `pymodalib.algorithms.wavelet` may be imported as `algorithms.wavelet`. This can cause some confusing errors.

For reliable uninstall behaviour, don't run `python setup.py install`. The command listed above, using `pip`, is more reliable. To install PyMODAlib from source without using editable mode, you can run `pip install .`.

Don't install the library from source if your terminal has navigated to the folder via a symlink. 

### MATLAB-packaged libraries

Currently, the MATLAB-packaged libraries are not required if functionality that depends on them is not used. MATLAB-packaged libraries are installed by downloading PyMODA and installing its dependencies.

Functions that require the MATLAB Runtime will be marked with the `matlabwrapper` decorator. This will check if the correct version of the MATLAB Runtime is installed.

> **Note:** MATLAB libraries are still incompatible with Python 3.8. When MATLAB R2020a releases, Python 3.8 support will be added but the current required version of the MATLAB Runtime will no longer be supported (users will need to upgrade to the newer Runtime).

### Packaging the project for PyPI

This section describes how to publish an update to PyPI. 

From the [documentation](https://packaging.python.org/guides/distributing-packages-using-setuptools/#packaging-your-project):

```bash
rm -r dist/
python setup.py sdist
python setup.py bdist_wheel
twine upload dist/*
```
