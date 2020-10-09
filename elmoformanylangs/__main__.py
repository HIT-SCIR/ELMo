#!/usr/bin/env python
from __future__ import print_function
from __future__ import unicode_literals
import os
import sys
import codecs
import argparse
import logging
import json
import torch
from .modules.embedding_layer import EmbeddingLayer
from .utils import dict2namedtuple
from .frontend import Model
from .frontend import create_batches
import numpy as np
import h5py

logger = logging.getLogger('elmoformanylangs')


def read_corpus(path, max_chars=None):
  """
  read raw text file. The format of the input is like, one sentence per line
  words are separated by '\t'

  :param path:
  :param max_chars: int, the number of maximum characters in a word, this
    parameter is used when the model is configured with CNN word encoder.
  :return:
  """
  dataset = []
  textset = []
  with codecs.open(path, 'r', encoding='utf-8') as fin:
    for line in fin.read().strip().split('\n'):
      data = ['<bos>']
      text = []
      for token in line.split('\t'):
        text.append(token)
        if max_chars is not None and len(token) + 2 > max_chars:
          token = token[:max_chars - 2]
        data.append(token)
      data.append('<eos>')
      dataset.append(data)
      textset.append(text)
  return dataset, textset


def read_conll_corpus(path, max_chars=None):
  """
  read text in CoNLL-U format.

  :param path:
  :param max_chars:
  :return:
  """
  dataset = []
  textset = []
  with codecs.open(path, 'r', encoding='utf-8') as fin:
    for payload in fin.read().strip().split('\n\n'):
      data = ['<bos>']
      text = []
      lines = payload.splitlines()
      body = [line for line in lines if not line.startswith('#')]
      for line in body:
        fields = line.split('\t')
        num, token = fields[0], fields[1]
        if '-' in num or '.' in num:
          continue
        text.append(token)
        if max_chars is not None and len(token) + 2 > max_chars:
          token = token[:max_chars - 2]
        data.append(token)
      data.append('<eos>')
      dataset.append(data)
      textset.append(text)
  return dataset, textset


def read_conll_char_corpus(path, max_chars=None):
  """

  :param path:
  :param max_chars:
  :return:
  """
  dataset = []
  textset = []
  with codecs.open(path, 'r', encoding='utf-8') as fin:
    for payload in fin.read().strip().split('\n\n'):
      data = ['<bos>']
      text = []
      lines = payload.splitlines()
      body = [line for line in lines if not line.startswith('#')]
      for line in body:
        fields = line.split('\t')
        num, token = fields[0], fields[1]
        if '-' in num or '.' in num:
          continue
        for ch in token:
          text.append(ch)
          if max_chars is not None and len(ch) + 2 > max_chars:
            ch = ch[:max_chars - 2]
          data.append(ch)
      data.append('<eos>')
      dataset.append(data)
      textset.append(text)
  return dataset, textset


def read_conll_char_vi_corpus(path, max_chars=None):
  """

  :param path:
  :param max_chars:
  :return:
  """
  dataset = []
  textset = []
  with codecs.open(path, 'r', encoding='utf-8') as fin:
    for payload in fin.read().strip().split('\n\n'):
      data = ['<bos>']
      text = []
      lines = payload.splitlines()
      body = [line for line in lines if not line.startswith('#')]
      for line in body:
        fields = line.split('\t')
        num, token = fields[0], fields[1]
        if '-' in num or '.' in num:
          continue
        for ch in token.split():
          text.append(ch)
          if max_chars is not None and len(ch) + 2 > max_chars:
            ch = ch[:max_chars - 2]
          data.append(ch)
      data.append('<eos>')
      dataset.append(data)
      textset.append(text)
  return dataset, textset


def test_main():
  # Configurations
  cmd = argparse.ArgumentParser('The testing components of')
  cmd.add_argument('--gpu', default=-1, type=int, help='use id of gpu, -1 if cpu.')
  cmd.add_argument('--input_format', default='plain', choices=('plain', 'conll', 'conll_char', 'conll_char_vi'),
                   help='the input format.')
  cmd.add_argument("--input", help="the path to the raw text file.")
  cmd.add_argument("--output_format", default='hdf5', help='the output format. Supported format includes (hdf5, txt).'
                                                           ' Use comma to separate the format identifiers,'
                                                           ' like \'--output_format=hdf5,plain\'')
  cmd.add_argument("--output_prefix", help='the prefix of the output file. The output file is in the format of '
                                           '<output_prefix>.<output_layer>.<output_format>')
  cmd.add_argument("--output_layer", help='the target layer to output. 0 for the word encoder, 1 for the first LSTM '
                                          'hidden layer, 2 for the second LSTM hidden layer, -1 for an average'
                                          'of 3 layers.')
  cmd.add_argument("--model", required=True, help="the path to the model.")
  cmd.add_argument("--batch_size", "--batch", type=int, default=1, help='the batch size.')
  args = cmd.parse_args(sys.argv[2:])

  if args.gpu >= 0:
    torch.cuda.set_device(args.gpu)
  use_cuda = args.gpu >= 0 and torch.cuda.is_available()
  # load the model configurations
  args2 = dict2namedtuple(json.load(codecs.open(os.path.join(args.model, 'config.json'), 'r', encoding='utf-8')))

  with open(os.path.join(args.model, args2.config_path), 'r') as fin:
    config = json.load(fin)

  # For the model trained with character-based word encoder.
  if config['token_embedder']['char_dim'] > 0:
    char_lexicon = {}
    with codecs.open(os.path.join(args.model, 'char.dic'), 'r', encoding='utf-8') as fpi:
      for line in fpi:
        tokens = line.strip().split('\t')
        if len(tokens) == 1:
          tokens.insert(0, '\u3000')
        token, i = tokens
        char_lexicon[token] = int(i)
    char_emb_layer = EmbeddingLayer(config['token_embedder']['char_dim'], char_lexicon, fix_emb=False, embs=None)
    logger.info('char embedding size: ' + str(len(char_emb_layer.word2id)))
  else:
    char_lexicon = None
    char_emb_layer = None

  # For the model trained with word form word encoder.
  if config['token_embedder']['word_dim'] > 0:
    word_lexicon = {}
    with codecs.open(os.path.join(args.model, 'word.dic'), 'r', encoding='utf-8') as fpi:
      for line in fpi:
        tokens = line.strip().split('\t')
        if len(tokens) == 1:
          tokens.insert(0, '\u3000')
        token, i = tokens
        word_lexicon[token] = int(i)
    word_emb_layer = EmbeddingLayer(config['token_embedder']['word_dim'], word_lexicon, fix_emb=False, embs=None)
    logger.info('word embedding size: ' + str(len(word_emb_layer.word2id)))
  else:
    word_lexicon = None
    word_emb_layer = None

  # instantiate the model
  model = Model(config, word_emb_layer, char_emb_layer, use_cuda)

  if use_cuda:
    model.cuda()

  logger.info(str(model))
  model.load_model(args.model)

  # read test data according to input format
  read_function = read_corpus if args.input_format == 'plain' else (
    read_conll_corpus if args.input_format == 'conll' else (
      read_conll_char_corpus if args.input_format == 'conll_char' else read_conll_char_vi_corpus))

  if config['token_embedder']['name'].lower() == 'cnn':
    test, text = read_function(args.input, config['token_embedder']['max_characters_per_token'])
  else:
    test, text = read_function(args.input)

  # create test batches from the input data.
  test_w, test_c, test_lens, test_masks, test_text = create_batches(
    test, args.batch_size, word_lexicon, char_lexicon, config, text=text)

  # configure the model to evaluation mode.
  model.eval()

  sent_set = set()
  cnt = 0

  output_formats = args.output_format.split(',')
  output_layers = map(int, args.output_layer.split(','))

  handlers = {}
  for output_format in output_formats:
    if output_format not in ('hdf5', 'txt'):
      print('Unknown output_format: {0}'.format(output_format))
      continue
    for output_layer in output_layers:
      filename = '{0}.ly{1}.{2}'.format(args.output_prefix, output_layer, output_format)
      handlers[output_format, output_layer] = \
        h5py.File(filename, 'w') if output_format == 'hdf5' else open(filename, 'w')

  for w, c, lens, masks, texts in zip(test_w, test_c, test_lens, test_masks, test_text):
    output = model.forward(w, c, masks)
    for i, text in enumerate(texts):
      sent = '\t'.join(text)
      sent = sent.replace('.', '$period$')
      sent = sent.replace('/', '$backslash$')
      if sent in sent_set:
        continue
      sent_set.add(sent)
      if config['encoder']['name'].lower() == 'lstm':
        data = output[i, 1:lens[i]-1, :].data
        if use_cuda:
          data = data.cpu()
        data = data.numpy()
      elif config['encoder']['name'].lower() == 'elmo':
        data = output[:, i, 1:lens[i]-1, :].data
        if use_cuda:
          data = data.cpu()
        data = data.numpy()

      for (output_format, output_layer) in handlers:
        fout = handlers[output_format, output_layer]
        if output_layer == -1:
          payload = np.average(data, axis=0)
        else:
          payload = data[output_layer]
        if output_format == 'hdf5':
          fout.create_dataset(sent, payload.shape, dtype='float32', data=payload)
        else:
          for word, row in zip(text, payload):
            print('{0}\t{1}'.format(word, '\t'.join(['{0:.8f}'.format(elem) for elem in row])), file=fout)
          print('', file=fout)

      cnt += 1
      if cnt % 1000 == 0:
        logger.info('Finished {0} sentences.'.format(cnt))
  for _, handler in handlers.items():
    handler.close()


if __name__ == "__main__":
  if len(sys.argv) > 1 and sys.argv[1] == 'test':
    test_main()
  else:
    print('Usage: {0} [test] [options]'.format(sys.argv[0]), file=sys.stderr)
