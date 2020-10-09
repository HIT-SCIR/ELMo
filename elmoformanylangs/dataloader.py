#!/usr/bin/env python
from __future__ import unicode_literals
import codecs
import numpy as np


def pad(sequences, pad_token='<pad>', pad_left=False):
  """
  input sequences is a list of text sequence [[str]]
  pad each text sequence to the length of the longest

  :param sequences:
  :param pad_token:
  :param pad_left:
  :return:
  """
  # max_len = max(5,max(len(seq) for seq in sequences))
  max_len = max(len(seq) for seq in sequences)
  if pad_left:
    return [[pad_token]*(max_len-len(seq)) + seq for seq in sequences]
  return [seq + [pad_token]*(max_len-len(seq)) for seq in sequences]


def load_embedding_npz(path):
  data = np.load(path)
  return [str(w) for w in data['words']], data['vals']


def load_embedding_txt(path):
  words = []
  vals = []
  with codecs.open(path, 'r', encoding='utf-8') as fin:
    fin.readline()
    for line in fin:
      line = line.strip()
      if line:
        parts = line.split()
        words.append(parts[0])
        vals += [float(x) for x in parts[1:]]  # equal to append
  return words, np.asarray(vals).reshape(len(words), -1)  # reshape


def load_embedding(path):
  if path.endswith(".npz"):
    return load_embedding_npz(path)
  else:
    return load_embedding_txt(path)
