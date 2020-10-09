#!/usr/bin/env python
from __future__ import unicode_literals
import collections
import itertools


def flatten(lst):
  return list(itertools.chain.from_iterable(lst))


def deep_iter(x):
  if isinstance(x, list) or isinstance(x, tuple):
    for u in x:
      for v in deep_iter(u):
        yield v
  else:
    yield


def dict2namedtuple(dic):
  return collections.namedtuple('Namespace', dic.keys())(**dic)
