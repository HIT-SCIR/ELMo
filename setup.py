#!/usr/bin/env python
import setuptools


setuptools.setup(
  name="elmoformanylangs",
  version="0.0.4",
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
  url="https://github.com/HIT-SCIR/ELMoForManyLangs",
  classifiers=[
    "Development Status :: 3 - Alpha",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3.6",
  ],
)
