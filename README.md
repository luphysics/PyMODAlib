# PyMODAlib

*Current status: experimental and work-in-progress*.

`PyMODAlib` is a Python library containing the algorithms used by PyMODA. Some of these algorithms are MATLAB-packaged libraries; some are pure-Python translations.

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
