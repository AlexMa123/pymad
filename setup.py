from setuptools import setup, find_packages
from pymad.mad import cc

setup(
    name='pymad',
    version='0.1.0',
    author='Yaopeng Ma',
    packages=find_packages(),
    ext_modules=[cc.distutils_extension()],
)
