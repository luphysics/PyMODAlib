# PyMODAlib

`PyMODAlib` is a Python library containing the algorithms used by PyMODA. 

Some of these algorithms are MATLAB-packaged libraries, while some are Python translations of MODA's algorithms.

## Installing PyMODAlib

`PyMODAlib` can be installed using `pip`:

```bash
pip install PyMODAlib
```

## Current status

`PyMODAlib` is still early in development. Currently, the features implemented are:

- Detecting harmonics.

## Examples

There are examples of using PyMODAlib's functions in the `examples` directory.

## Developer notes

### Developing PyMODAlib

When developing PyMODAlib, you can test your changes by installing the library locally. From the root of the repository, run:

```bash
python setup.py install
```

### Packaging the project

From the [documentation](https://packaging.python.org/guides/distributing-packages-using-setuptools/#packaging-your-project):

```bash
rm -r dist/
python setup.py sdist
python setup.py bdist_wheel
twine upload dist/*
```
