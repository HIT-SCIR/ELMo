#!/usr/bin/env python
from __future__ import print_function
from __future__ import unicode_literals
import os
import errno
import sys
import codecs
import argparse
import time
import random
import logging
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from .modules.elmo import ElmobiLm
from .modules.lstm import LstmbiLm
from .modules.token_embedder import ConvTokenEmbedder, LstmTokenEmbedder
from .modules.embedding_layer import EmbeddingLayer
from .modules.classify_layer import SoftmaxLayer, CNNSoftmaxLayer, SampledSoftmaxLayer
from .dataloader import load_embedding
from .utils import dict2namedtuple
from collections import Counter
import numpy as np

logger = logging.getLogger('elmoformanylangs')


def divide(data, valid_size):
  valid_size = min(valid_size, len(data) // 10)
  random.shuffle(data)
  return data[valid_size:], data[:valid_size]


def break_sentence(sentence, max_sent_len):
  """
  For example, for a sentence with 70 words, supposing the the `max_sent_len'
  is 30, break it into 3 sentences.

  :param sentence: list[str] the sentence
  :param max_sent_len:
  :return:
  """
  ret = []
  cur = 0
  length = len(sentence)
  while cur < length:
    if cur + max_sent_len + 5 >= length:
      ret.append(sentence[cur: length])
      break
    ret.append(sentence[cur: min(length, cur + max_sent_len)])
    cur += max_sent_len
  return ret


def read_corpus(path, max_chars=None, max_sent_len=20):
  """
  read raw text file
  :param path: str
  :param max_chars: int
  :param max_sent_len: int
  :return:
  """
  data = []
  with codecs.open(path, 'r', encoding='utf-8') as fin:
    for line in fin:
      data.append('<bos>')
      for token in line.strip().split():
        if max_chars is not None and len(token) + 2 > max_chars:
          token = token[:max_chars - 2]
        data.append(token)
      data.append('<eos>')
  dataset = break_sentence(data, max_sent_len)
  return dataset


def create_one_batch(x, word2id, char2id, config, oov='<oov>', pad='<pad>', sort=True):
  """

  :param x:
  :param word2id: dict
  :param char2id: dict
  :param config:
  :param oov:
  :param pad:
  :param sort:
  :return:
  """
  batch_size = len(x)
  lst = list(range(batch_size))
  if sort:
    lst.sort(key=lambda l: -len(x[l]))

  x = [x[i] for i in lst]
  lens = [len(x[i]) for i in lst]
  max_len = max(lens)

  if word2id is not None:
    oov_id, pad_id = word2id.get(oov, None), word2id.get(pad, None)
    assert oov_id is not None and pad_id is not None
    batch_w = torch.LongTensor(batch_size, max_len).fill_(pad_id)
    for i, x_i in enumerate(x):
      for j, x_ij in enumerate(x_i):
        batch_w[i][j] = word2id.get(x_ij, oov_id)
  else:
    batch_w = None

  if char2id is not None:
    bow_id, eow_id, oov_id, pad_id = char2id.get('<eow>', None), char2id.get('<bow>', None), char2id.get(oov, None), char2id.get(pad, None)

    assert bow_id is not None and eow_id is not None and oov_id is not None and pad_id is not None

    if config['token_embedder']['name'].lower() == 'cnn':
      max_chars = config['token_embedder']['max_characters_per_token']
      assert max([len(w) for i in lst for w in x[i]]) + 2 <= max_chars
    elif config['token_embedder']['name'].lower() == 'lstm':
      max_chars = max([len(w) for i in lst for w in x[i]]) + 2  # counting the <bow> and <eow>

    batch_c = torch.LongTensor(batch_size, max_len, max_chars).fill_(pad_id)

    for i, x_i in enumerate(x):
      for j, x_ij in enumerate(x_i):
        batch_c[i][j][0] = bow_id
        if x_ij == '<bos>' or x_ij == '<eos>':
          batch_c[i][j][1] = char2id.get(x_ij)
          batch_c[i][j][2] = eow_id
        else:
          for k, c in enumerate(x_ij):
            batch_c[i][j][k + 1] = char2id.get(c, oov_id)
          batch_c[i][j][len(x_ij) + 1] = eow_id
  else:
    batch_c = None

  masks = [torch.LongTensor(batch_size, max_len).fill_(0), [], []]

  for i, x_i in enumerate(x):
    for j in range(len(x_i)):
      masks[0][i][j] = 1
      if j + 1 < len(x_i):
        masks[1].append(i * max_len + j)
      if j > 0: 
        masks[2].append(i * max_len + j)

  assert len(masks[1]) <= batch_size * max_len
  assert len(masks[2]) <= batch_size * max_len

  masks[1] = torch.LongTensor(masks[1])
  masks[2] = torch.LongTensor(masks[2])

  return batch_w, batch_c, lens, masks


# shuffle training examples and create mini-batches
def create_batches(x, batch_size, word2id, char2id, config, perm=None, shuffle=True, sort=True, use_cuda=False):
  """

  :param x:
  :param batch_size:
  :param word2id:
  :param char2id:
  :param config:
  :param perm:
  :param shuffle:
  :param sort:
  :param use_cuda:
  :return:
  """
  lst = perm or list(range(len(x)))
  if shuffle:
    random.shuffle(lst)

  if sort:
    lst.sort(key=lambda l: -len(x[l]))

  x = [x[i] for i in lst]

  sum_len = 0.0
  batches_w, batches_c, batches_lens, batches_masks = [], [], [], []
  size = batch_size
  nbatch = (len(x) - 1) // size + 1
  for i in range(nbatch):
    start_id, end_id = i * size, (i + 1) * size
    bw, bc, blens, bmasks = create_one_batch(x[start_id: end_id], word2id, char2id, config, sort=sort)
    sum_len += sum(blens)
    batches_w.append(bw)
    batches_c.append(bc)
    batches_lens.append(blens)
    batches_masks.append(bmasks)

  if sort:
    perm = list(range(nbatch))
    random.shuffle(perm)
    batches_w = [batches_w[i] for i in perm]
    batches_c = [batches_c[i] for i in perm]
    batches_lens = [batches_lens[i] for i in perm]
    batches_masks = [batches_masks[i] for i in perm]

  logger.info("{} batches, avg len: {:.1f}".format(nbatch, sum_len / len(x)))
  return batches_w, batches_c, batches_lens, batches_masks


class Model(nn.Module):
  def __init__(self, config, word_emb_layer, char_emb_layer, n_class, use_cuda=False):
    super(Model, self).__init__() 
    self.use_cuda = use_cuda
    self.config = config

    if config['token_embedder']['name'].lower() == 'cnn':
      self.token_embedder = ConvTokenEmbedder(config, word_emb_layer, char_emb_layer, use_cuda)
    elif config['token_embedder']['name'].lower() == 'lstm':
      self.token_embedder = LstmTokenEmbedder(config, word_emb_layer, char_emb_layer, use_cuda)

    if config['encoder']['name'].lower() == 'elmo':
      self.encoder = ElmobiLm(config, use_cuda)
    elif config['encoder']['name'].lower() == 'lstm':
      self.encoder = LstmbiLm(config, use_cuda)

    self.output_dim = config['encoder']['projection_dim']
    if config['classifier']['name'].lower() == 'softmax':
      self.classify_layer = SoftmaxLayer(self.output_dim, n_class)
    elif config['classifier']['name'].lower() == 'cnn_softmax':
      self.classify_layer = CNNSoftmaxLayer(self.token_embedder, self.output_dim, n_class,
                                            config['classifier']['n_samples'], config['classifier']['corr_dim'],
                                            use_cuda)
    elif config['classifier']['name'].lower() == 'sampled_softmax':
      self.classify_layer = SampledSoftmaxLayer(self.output_dim, n_class, config['classifier']['n_samples'], use_cuda)

  def forward(self, word_inp, chars_inp, mask_package):
    """

    :param word_inp:
    :param chars_inp:
    :param mask_package: Tuple[]
    :return:
    """
    classifier_name = self.config['classifier']['name'].lower()

    if self.training and classifier_name == 'cnn_softmax' or classifier_name == 'sampled_softmax':
      self.classify_layer.update_negative_samples(word_inp, chars_inp, mask_package[0])
      self.classify_layer.update_embedding_matrix()

    token_embedding = self.token_embedder(word_inp, chars_inp, (mask_package[0].size(0), mask_package[0].size(1)))
    token_embedding = F.dropout(token_embedding, self.config['dropout'], self.training)

    encoder_name = self.config['encoder']['name'].lower()
    if encoder_name == 'elmo':
      mask = Variable(mask_package[0].cuda()).cuda() if self.use_cuda else Variable(mask_package[0])
      encoder_output = self.encoder(token_embedding, mask)
      encoder_output = encoder_output[1]
      # [batch_size, len, hidden_size]
    elif encoder_name == 'lstm':
      encoder_output = self.encoder(token_embedding)
    else:
      raise ValueError('')

    encoder_output = F.dropout(encoder_output, self.config['dropout'], self.training)
    forward, backward = encoder_output.split(self.output_dim, 2)

    word_inp = Variable(word_inp)
    if self.use_cuda:
      word_inp = word_inp.cuda()

    mask1 = Variable(mask_package[1].cuda()).cuda() if self.use_cuda else Variable(mask_package[1])
    mask2 = Variable(mask_package[2].cuda()).cuda() if self.use_cuda else Variable(mask_package[2])

    forward_x = forward.contiguous().view(-1, self.output_dim).index_select(0, mask1)
    forward_y = word_inp.contiguous().view(-1).index_select(0, mask2)

    backward_x = backward.contiguous().view(-1, self.output_dim).index_select(0, mask2)
    backward_y = word_inp.contiguous().view(-1).index_select(0, mask1)

    return self.classify_layer(forward_x, forward_y), self.classify_layer(backward_x, backward_y)

  def save_model(self, path, save_classify_layer):
    torch.save(self.token_embedder.state_dict(), os.path.join(path, 'token_embedder.pkl'))    
    torch.save(self.encoder.state_dict(), os.path.join(path, 'encoder.pkl'))
    if save_classify_layer:
      torch.save(self.classify_layer.state_dict(), os.path.join(path, 'classifier.pkl'))

  def load_model(self, path):
    self.token_embedder.load_state_dict(torch.load(os.path.join(path, 'token_embedder.pkl')))
    self.encoder.load_state_dict(torch.load(os.path.join(path, 'encoder.pkl')))
    self.classify_layer.load_state_dict(torch.load(os.path.join(path, 'classifier.pkl')))


def eval_model(model, valid):
  model.eval()
  if model.config['classifier']['name'].lower() == 'cnn_softmax' or \
      model.config['classifier']['name'].lower() == 'sampled_softmax':
    model.classify_layer.update_embedding_matrix()
  total_loss, total_tag = 0.0, 0
  valid_w, valid_c, valid_lens, valid_masks = valid
  for w, c, lens, masks in zip(valid_w, valid_c, valid_lens, valid_masks):
    loss_forward, loss_backward = model.forward(w, c, masks)
    total_loss += loss_forward.data[0]
    n_tags = sum(lens)
    total_tag += n_tags
  model.train()
  return np.exp(total_loss / total_tag)


def train_model(epoch, opt, model, optimizer,
                train, valid, test, best_train, best_valid, test_result):
  """
  Training model for one epoch

  :param epoch:
  :param opt:
  :param model:
  :param optimizer:
  :param train:
  :param best_train:
  :param valid:
  :param best_valid:
  :param test:
  :param test_result:
  :return:
  """
  model.train()

  total_loss, total_tag = 0.0, 0
  cnt = 0
  start_time = time.time()

  train_w, train_c, train_lens, train_masks = train

  lst = list(range(len(train_w)))
  random.shuffle(lst)
  
  train_w = [train_w[l] for l in lst]
  train_c = [train_c[l] for l in lst]
  train_lens = [train_lens[l] for l in lst]
  train_masks = [train_masks[l] for l in lst]

  for w, c, lens, masks in zip(train_w, train_c, train_lens, train_masks):
    cnt += 1
    model.zero_grad()
    loss_forward, loss_backward = model.forward(w, c, masks)

    loss = (loss_forward + loss_backward) / 2.0
    total_loss += loss_forward.data[0]
    n_tags = sum(lens)
    total_tag += n_tags
    loss.backward()

    torch.nn.utils.clip_grad_norm(model.parameters(), opt.clip_grad)
    optimizer.step()
    if cnt * opt.batch_size % 1024 == 0:
      logger.info("Epoch={} iter={} lr={:.6f} train_ppl={:.6f} time={:.2f}s".format(
        epoch, cnt, optimizer.param_groups[0]['lr'],
        np.exp(total_loss / total_tag), time.time() - start_time
      ))
      start_time = time.time()

    if cnt % opt.eval_steps == 0 or cnt % len(train_w) == 0:
      if valid is None:
        train_ppl = np.exp(total_loss / total_tag)
        logger.info("Epoch={} iter={} lr={:.6f} train_ppl={:.6f}".format(
          epoch, cnt, optimizer.param_groups[0]['lr'], train_ppl))
        if train_ppl < best_train:
          best_train = train_ppl
          logger.info("New record achieved on training dataset!")
          model.save_model(opt.model, opt.save_classify_layer)      
      else:
        valid_ppl = eval_model(model, valid)
        logger.info("Epoch={} iter={} lr={:.6f} valid_ppl={:.6f}".format(
          epoch, cnt, optimizer.param_groups[0]['lr'], valid_ppl))

        if valid_ppl < best_valid:
          model.save_model(opt.model, opt.save_classify_layer)
          best_valid = valid_ppl
          logger.info("New record achieved!")

          if test is not None:
            test_result = eval_model(model, test)
            logger.info("Epoch={} iter={} lr={:.6f} test_ppl={:.6f}".format(
              epoch, cnt, optimizer.param_groups[0]['lr'], test_result))
  return best_train, best_valid, test_result


def get_truncated_vocab(dataset, min_count):
  """

  :param dataset:
  :param min_count: int
  :return:
  """
  word_count = Counter()
  for sentence in dataset:
    word_count.update(sentence)

  word_count = list(word_count.items())
  word_count.sort(key=lambda x: x[1], reverse=True)

  i = 0
  for word, count in word_count:
    if count < min_count:
      break
    i += 1

  logger.info('Truncated word count: {0}.'.format(sum([count for word, count in word_count[i:]])))
  logger.info('Original vocabulary size: {0}.'.format(len(word_count)))
  return word_count[:i]


def train():
  cmd = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')
  cmd.add_argument('--seed', default=1, type=int, help='The random seed.')
  cmd.add_argument('--gpu', default=-1, type=int, help='Use id of gpu, -1 if cpu.')

  cmd.add_argument('--train_path', required=True, help='The path to the training file.')
  cmd.add_argument('--valid_path', help='The path to the development file.')
  cmd.add_argument('--test_path', help='The path to the testing file.')

  cmd.add_argument('--config_path', required=True, help='the path to the config file.')
  cmd.add_argument("--word_embedding", help="The path to word vectors.")

  cmd.add_argument('--optimizer', default='sgd', choices=['sgd', 'adam', 'adagrad'],
                   help='the type of optimizer: valid options=[sgd, adam, adagrad]')
  cmd.add_argument("--lr", type=float, default=0.01, help='the learning rate.')
  cmd.add_argument("--lr_decay", type=float, default=0, help='the learning rate decay.')

  cmd.add_argument("--model", required=True, help="path to save model")
  
  cmd.add_argument("--batch_size", "--batch", type=int, default=32, help='the batch size.')
  cmd.add_argument("--max_epoch", type=int, default=100, help='the maximum number of iteration.')
  
  cmd.add_argument("--clip_grad", type=float, default=5, help='the tense of clipped grad.')

  cmd.add_argument('--max_sent_len', type=int, default=20, help='maximum sentence length.')

  cmd.add_argument('--min_count', type=int, default=5, help='minimum word count.')

  cmd.add_argument('--max_vocab_size', type=int, default=150000, help='maximum vocabulary size.')

  cmd.add_argument('--save_classify_layer', default=False, action='store_true',
                   help="whether to save the classify layer")

  cmd.add_argument('--valid_size', type=int, default=0, help="size of validation dataset when there's no valid.")
  cmd.add_argument('--eval_steps', required=False, type=int, help='report every xx batches.')

  opt = cmd.parse_args(sys.argv[2:])

  with open(opt.config_path, 'r') as fin:
    config = json.load(fin)

  # Dump configurations
  print(opt)
  print(config)

  # set seed.
  torch.manual_seed(opt.seed)
  random.seed(opt.seed)
  if opt.gpu >= 0:
    torch.cuda.set_device(opt.gpu)
    if opt.seed > 0:
      torch.cuda.manual_seed(opt.seed)

  use_cuda = opt.gpu >= 0 and torch.cuda.is_available()

  token_embedder_name = config['token_embedder']['name'].lower()
  token_embedder_max_chars = config['token_embedder'].get('max_characters_per_token', None)
  if token_embedder_name == 'cnn':
    train_data = read_corpus(opt.train_path, token_embedder_max_chars, opt.max_sent_len)
  elif token_embedder_name == 'lstm':
    train_data = read_corpus(opt.train_path, opt.max_sent_len)
  else:
    raise ValueError('Unknown token embedder name: {}'.format(token_embedder_name))

  logger.info('training instance: {}, training tokens: {}.'.format(len(train_data),
                                                                    sum([len(s) - 1 for s in train_data])))

  if opt.valid_path is not None:
    if token_embedder_name == 'cnn':
      valid_data = read_corpus(opt.valid_path, token_embedder_max_chars, opt.max_sent_len)
    elif token_embedder_name == 'lstm':
      valid_data = read_corpus(opt.valid_path, opt.max_sent_len)
    else:
      raise ValueError('Unknown token embedder name: {}'.format(token_embedder_name))
    logger.info('valid instance: {}, valid tokens: {}.'.format(len(valid_data),
                                                                sum([len(s) - 1 for s in valid_data])))
  elif opt.valid_size > 0:
    train_data, valid_data = divide(train_data, opt.valid_size)
    logger.info('training instance: {}, training tokens after division: {}.'.format(
      len(train_data), sum([len(s) - 1 for s in train_data])))
    logger.info('valid instance: {}, valid tokens: {}.'.format(
      len(valid_data), sum([len(s) - 1 for s in valid_data])))
  else:
    valid_data = None

  if opt.test_path is not None:
    if token_embedder_name == 'cnn':
      test_data = read_corpus(opt.test_path, token_embedder_max_chars, opt.max_sent_len)
    elif token_embedder_name == 'lstm':
      test_data = read_corpus(opt.test_path, opt.max_sent_len)
    else:
      raise ValueError('Unknown token embedder name: {}'.format(token_embedder_name))
    logger.info('testing instance: {}, testing tokens: {}.'.format(
      len(test_data), sum([len(s) - 1 for s in test_data])))
  else:
    test_data = None

  if opt.word_embedding is not None:
    embs = load_embedding(opt.word_embedding)
    word_lexicon = {word: i for i, word in enumerate(embs[0])}  
  else:
    embs = None
    word_lexicon = {}

  # Maintain the vocabulary. vocabulary is used in either WordEmbeddingInput or softmax classification
  vocab = get_truncated_vocab(train_data, opt.min_count)

  # Ensure index of '<oov>' is 0
  for special_word in ['<oov>', '<bos>', '<eos>',  '<pad>']:
    if special_word not in word_lexicon:
      word_lexicon[special_word] = len(word_lexicon)

  for word, _ in vocab:
    if word not in word_lexicon:
      word_lexicon[word] = len(word_lexicon)

  # Word Embedding
  if config['token_embedder']['word_dim'] > 0:
    word_emb_layer = EmbeddingLayer(config['token_embedder']['word_dim'], word_lexicon, fix_emb=False, embs=embs)
    logger.info('Word embedding size: {0}'.format(len(word_emb_layer.word2id)))
  else:
    word_emb_layer = None
    logger.info('Vocabulary size: {0}'.format(len(word_lexicon)))

  # Character Lexicon
  if config['token_embedder']['char_dim'] > 0:
    char_lexicon = {}
    for sentence in train_data:
      for word in sentence:
        for ch in word:
          if ch not in char_lexicon:
            char_lexicon[ch] = len(char_lexicon)

    for special_char in ['<bos>', '<eos>', '<oov>', '<pad>', '<bow>', '<eow>']:
      if special_char not in char_lexicon:
        char_lexicon[special_char] = len(char_lexicon)

    char_emb_layer = EmbeddingLayer(config['token_embedder']['char_dim'], char_lexicon, fix_emb=False)
    logger.info('Char embedding size: {0}'.format(len(char_emb_layer.word2id)))
  else:
    char_lexicon = None
    char_emb_layer = None

  train = create_batches(
    train_data, opt.batch_size, word_lexicon, char_lexicon, config, use_cuda=use_cuda)

  if opt.eval_steps is None:
    opt.eval_steps = len(train[0])
  logger.info('Evaluate every {0} batches.'.format(opt.eval_steps))

  if valid_data is not None:
    valid = create_batches(
      valid_data, opt.batch_size, word_lexicon, char_lexicon, config, sort=False, shuffle=False, use_cuda=use_cuda)
  else:
    valid = None

  if test_data is not None:
    test = create_batches(
      test_data, opt.batch_size, word_lexicon, char_lexicon, config, sort=False, shuffle=False, use_cuda=use_cuda)
  else:
    test = None

  label_to_ix = word_lexicon
  logger.info('vocab size: {0}'.format(len(label_to_ix)))
  
  nclasses = len(label_to_ix)

  model = Model(config, word_emb_layer, char_emb_layer, nclasses, use_cuda)
  logger.info(str(model))
  if use_cuda:
    model = model.cuda()

  need_grad = lambda x: x.requires_grad
  if opt.optimizer.lower() == 'adam':
    optimizer = optim.Adam(filter(need_grad, model.parameters()), lr=opt.lr)
  elif opt.optimizer.lower() == 'sgd':
    optimizer = optim.SGD(filter(need_grad, model.parameters()), lr=opt.lr)
  elif opt.optimizer.lower() == 'adagrad':
    optimizer = optim.Adagrad(filter(need_grad, model.parameters()), lr=opt.lr)
  else:
    raise ValueError('Unknown optimizer {}'.format(opt.optimizer.lower()))

  try:
    os.makedirs(opt.model)
  except OSError as exception:
    if exception.errno != errno.EEXIST:
      raise

  if config['token_embedder']['char_dim'] > 0:
    with codecs.open(os.path.join(opt.model, 'char.dic'), 'w', encoding='utf-8') as fpo:
      for ch, i in char_emb_layer.word2id.items():
        print('{0}\t{1}'.format(ch, i), file=fpo)

  with codecs.open(os.path.join(opt.model, 'word.dic'), 'w', encoding='utf-8') as fpo:
    for w, i in word_lexicon.items():
      print('{0}\t{1}'.format(w, i), file=fpo)

  json.dump(vars(opt), codecs.open(os.path.join(opt.model, 'config.json'), 'w', encoding='utf-8'))

  best_train = 1e+8
  best_valid = 1e+8
  test_result = 1e+8

  for epoch in range(opt.max_epoch):
    best_train, best_valid, test_result = train_model(epoch, opt, model, optimizer,
                                                      train, valid, test, best_train, best_valid, test_result)
    if opt.lr_decay > 0:
      optimizer.param_groups[0]['lr'] *= opt.lr_decay

  if valid_data is None:
    logger.info("best train ppl: {:.6f}.".format(best_train))
  elif test_data is None:
    logger.info("best train ppl: {:.6f}, best valid ppl: {:.6f}.".format(best_train, best_valid))
  else:
    logger.info("best train ppl: {:.6f}, best valid ppl: {:.6f}, test ppl: {:.6f}.".format(best_train, best_valid, test_result))


def test():
  cmd = argparse.ArgumentParser('The testing components of')
  cmd.add_argument('--gpu', default=-1, type=int, help='use id of gpu, -1 if cpu.')
  cmd.add_argument("--input", help="the path to the raw text file.")
  cmd.add_argument("--model", required=True, help="path to save model")
  cmd.add_argument("--batch_size", "--batch", type=int, default=1, help='the batch size.')
  args = cmd.parse_args(sys.argv[2:])

  if args.gpu >= 0:
    torch.cuda.set_device(args.gpu)
  use_cuda = args.gpu >= 0 and torch.cuda.is_available()
  
  args2 = dict2namedtuple(json.load(codecs.open(os.path.join(args.model, 'config.json'), 'r', encoding='utf-8')))

  with open(args2.config_path, 'r') as fin:
    config = json.load(fin)

  if config['token_embedder']['char_dim'] > 0:
    char_lexicon = {}
    with codecs.open(os.path.join(args.model, 'char.dic'), 'r', encoding='utf-8') as fpi:
      for line in fpi:
        tokens = line.strip().split('\t')
        if len(tokens) == 1:
          tokens.insert(0, '\u3000')
        token, i = tokens
        char_lexicon[token] = int(i)
    char_emb_layer = EmbeddingLayer(config['token_embedder']['char_dim'], char_lexicon, fix_emb=False)
    logger.info('char embedding size: ' + str(len(char_emb_layer.word2id)))
  else:
    char_lexicon = None
    char_emb_layer = None

  word_lexicon = {}
  with codecs.open(os.path.join(args.model, 'word.dic'), 'r', encoding='utf-8') as fpi:
    for line in fpi:
      tokens = line.strip().split('\t')
      if len(tokens) == 1:
        tokens.insert(0, '\u3000')
      token, i = tokens
      word_lexicon[token] = int(i)

  if config['token_embedder']['word_dim'] > 0:
    word_emb_layer = EmbeddingLayer(config['token_embedder']['word_dim'], word_lexicon, fix_emb=False, embs=None)
    logger.info('word embedding size: ' + str(len(word_emb_layer.word2id)))
  else:
    word_emb_layer = None
  
  model = Model(config, word_emb_layer, char_emb_layer, len(word_lexicon), use_cuda)

  if use_cuda:
    model.cuda()

  logger.info(str(model))
  model.load_model(args.model)
  if config['token_embedder']['name'].lower() == 'cnn':
    test = read_corpus(args.input, config['token_embedder']['max_characters_per_token'], max_sent_len=10000)
  elif config['token_embedder']['name'].lower() == 'lstm':
    test = read_corpus(args.input, max_sent_len=10000)
  else:
    raise ValueError('')

  test_w, test_c, test_lens, test_masks = create_batches(
    test, args.batch_size, word_lexicon, char_lexicon, config, sort=False, shuffle=False, use_cuda=use_cuda)

  test_result = eval_model(model, (test_w, test_c, test_lens, test_masks))

  logger.info("test_ppl={:.6f}".format(test_result))


if __name__ == "__main__":
  if len(sys.argv) > 1 and sys.argv[1] == 'train':
    train()
  elif len(sys.argv) > 1 and sys.argv[1] == 'test':
    test()
  else:
    print('Usage: {0} [train|test] [options]'.format(sys.argv[0]), file=sys.stderr)
