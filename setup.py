#!/usr/bin/env python
import setuptools


setuptools.setup(
  name="elmoformanylangs",
  version="0.0.3.post1",
  packages=setuptools.find_packages(),
  install_requires=[
    "torch",
    "h5py",
    "numpy",
    "overrides",
  ],
  package_data={'configs': ['elmoformanylangs/configs/*.json']},
  include_package_data=True,
  author="哈工大社会计算与信息检索研究中心",
  description="ELMo, updated to be usable with models for many languages",
  url="https://github.com/HIT-SCIR/ELMoForManyLangs",
  classifiers=[
    "Development Status :: 3 - Alpha",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3.6",
  ],
)
