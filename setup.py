#!/usr/bin/env python
import setuptools

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setuptools.setup(
  name="elmoformanylangs",
  version="0.0.4.post2",
  packages=setuptools.find_packages(),
  install_requires=[
    "torch",
    "h5py",
    "numpy",
    "overrides",
  ],
  package_data={'configs': ['elmoformanylangs/configs/*.json']},
  include_package_data=True,
  author="Research Center for Social Computing and Information Retrieval",
  description="ELMo, updated to be usable with models for many languages",
  long_description=long_description,
  long_description_content_type='text/markdown',
  url="https://github.com/HIT-SCIR/ELMoForManyLangs",
  classifiers=[
    "Development Status :: 3 - Alpha",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3.6",
  ],
)
