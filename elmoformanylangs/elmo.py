#!/usr/bin/env python
from __future__ import print_function
from __future__ import unicode_literals
import os
import codecs
import random
import logging
import json
import torch
from .modules.embedding_layer import EmbeddingLayer
from .utils import dict2namedtuple
from .frontend import create_one_batch
from .frontend import Model
import numpy as np

logger = logging.getLogger('elmoformanylangs')


def read_list(sents, max_chars=None):
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
    for sent in sents:
        data = ['<bos>']
        text = []
        for token in sent:
            text.append(token)
            if max_chars is not None and len(token) + 2 > max_chars:
                token = token[:max_chars - 2]
            data.append(token)
        data.append('<eos>')
        dataset.append(data)
        textset.append(text)
    return dataset, textset


def recover(li, ind):
    # li[piv], ind = torch.sort(li[piv], dim=0, descending=(not unsort))
    dummy = list(range(len(ind)))
    dummy.sort(key=lambda l: ind[l])
    li = [li[i] for i in dummy]
    return li


# shuffle training examples and create mini-batches
def create_batches(x, batch_size, word2id, char2id, config, perm=None, shuffle=False, sort=True, text=None):
    ind = list(range(len(x)))
    lst = perm or list(range(len(x)))
    if shuffle:
        random.shuffle(lst)

    if sort:
        lst.sort(key=lambda l: -len(x[l]))

    x = [x[i] for i in lst]
    ind = [ind[i] for i in lst]
    if text is not None:
        text = [text[i] for i in lst]

    sum_len = 0.0
    batches_w, batches_c, batches_lens, batches_masks, batches_text, batches_ind = [], [], [], [], [], []
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
        batches_ind.append(ind[start_id: end_id])
        if text is not None:
            batches_text.append(text[start_id: end_id])

    if sort:
        perm = list(range(nbatch))
        random.shuffle(perm)
        batches_w = [batches_w[i] for i in perm]
        batches_c = [batches_c[i] for i in perm]
        batches_lens = [batches_lens[i] for i in perm]
        batches_masks = [batches_masks[i] for i in perm]
        batches_ind = [batches_ind[i] for i in perm]
        if text is not None:
            batches_text = [batches_text[i] for i in perm]

    logger.info("{} batches, avg len: {:.1f}".format(
        nbatch, sum_len / len(x)))
    recover_ind = [item for sublist in batches_ind for item in sublist]
    if text is not None:
        return batches_w, batches_c, batches_lens, batches_masks, batches_text, recover_ind
    return batches_w, batches_c, batches_lens, batches_masks, recover_ind


class Embedder(object):
    def __init__(self, model_dir, batch_size=64):
        self.model_dir = model_dir
        self.model, self.config = self.get_model()
        self.batch_size = batch_size

    def get_model(self):
        # torch.cuda.set_device(1)
        self.use_cuda = torch.cuda.is_available()
        # load the model configurations
        args2 = dict2namedtuple(json.load(codecs.open(
            os.path.join(self.model_dir, 'config.json'), 'r', encoding='utf-8')))

        config_path = os.path.join(self.model_dir, args2.config_path)
        # Some of the available models may have the config in the
        # model dir, but the path given in the config directory was an
        # absolute path.
        if not os.path.exists(config_path):
            config_path = os.path.join(self.model_dir,
                                       os.path.split(config_path)[1])
            logger.warning("Could not find config.  Trying " + config_path)
        # In many cases, such as the publicly available English model,
        # the config is one of the default provided configs in
        # elmoformanylangs/configs
        if not os.path.exists(config_path):
            config_path = os.path.join(os.path.split(__file__)[0], "configs",
                                       os.path.split(config_path)[1])
            logger.warning("Could not find config.  Trying " + config_path)

        if not os.path.exists(config_path):
            raise FileNotFoundError("Could not find the model config in either the model directory "
                                    "or the default configs.  Path in config file: %s" % args2.config_path)

        with open(config_path, 'r') as fin:
            config = json.load(fin)

        # For the model trained with character-based word encoder.
        if config['token_embedder']['char_dim'] > 0:
            self.char_lexicon = {}
            with codecs.open(os.path.join(self.model_dir, 'char.dic'), 'r', encoding='utf-8') as fpi:
                for line in fpi:
                    tokens = line.strip().split('\t')
                    if len(tokens) == 1:
                        tokens.insert(0, '\u3000')
                    token, i = tokens
                    self.char_lexicon[token] = int(i)
            char_emb_layer = EmbeddingLayer(
                config['token_embedder']['char_dim'], self.char_lexicon, fix_emb=False, embs=None)
            logger.info('char embedding size: ' +
                        str(len(char_emb_layer.word2id)))
        else:
            self.char_lexicon = None
            char_emb_layer = None

        # For the model trained with word form word encoder.
        if config['token_embedder']['word_dim'] > 0:
            self.word_lexicon = {}
            with codecs.open(os.path.join(self.model_dir, 'word.dic'), 'r', encoding='utf-8') as fpi:
                for line in fpi:
                    tokens = line.strip().split('\t')
                    if len(tokens) == 1:
                        tokens.insert(0, '\u3000')
                    token, i = tokens
                    self.word_lexicon[token] = int(i)
            word_emb_layer = EmbeddingLayer(
                config['token_embedder']['word_dim'], self.word_lexicon, fix_emb=False, embs=None)
            logger.info('word embedding size: ' +
                        str(len(word_emb_layer.word2id)))
        else:
            self.word_lexicon = None
            word_emb_layer = None

        # instantiate the model
        model = Model(config, word_emb_layer, char_emb_layer, self.use_cuda)

        if self.use_cuda:
            model.cuda()

        logger.info(str(model))
        model.load_model(self.model_dir)

        # read test data according to input format

        # configure the model to evaluation mode.
        model.eval()
        return model, config

    def sents2elmo(self, sents, output_layer=-1):
        read_function = read_list

        if self.config['token_embedder']['name'].lower() == 'cnn':
            test, text = read_function(sents, self.config['token_embedder']['max_characters_per_token'])
        else:
            test, text = read_function(sents)

        # create test batches from the input data.
        test_w, test_c, test_lens, test_masks, test_text, recover_ind = create_batches(
            test, self.batch_size, self.word_lexicon, self.char_lexicon, self.config, text=text)

        cnt = 0

        after_elmo = []
        for w, c, lens, masks, texts in zip(test_w, test_c, test_lens, test_masks, test_text):
            output = self.model.forward(w, c, masks)
            for i, text in enumerate(texts):

                if self.config['encoder']['name'].lower() == 'lstm':
                    data = output[i, 1:lens[i]-1, :].data
                    if self.use_cuda:
                        data = data.cpu()
                    data = data.numpy()
                elif self.config['encoder']['name'].lower() == 'elmo':
                    data = output[:, i, 1:lens[i]-1, :].data
                    if self.use_cuda:
                        data = data.cpu()
                    data = data.numpy()

                if output_layer == -1:
                    payload = np.average(data, axis=0)
                elif output_layer == -2:
                    payload = data
                else:
                    payload = data[output_layer]
                after_elmo.append(payload)

                cnt += 1
                if cnt % 1000 == 0:
                    logger.info('Finished {0} sentences.'.format(cnt))

        after_elmo = recover(after_elmo, recover_ind)
        return after_elmo
