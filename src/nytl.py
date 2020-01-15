##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Albert Berenguel
## Computer Vision Center (CVC). Universitat Autonoma de Barcelona
## Email: aberenguel@cvc.uab.es
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##
## Arcknowledgments:
## https://github.com/ludc. Using some parts of his Omniglot code.
## https://github.com/AntreasAntoniou. Using some parts of his Omniglot code.
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import errno
import json


class NYT(data.Dataset):
    urls = [
        'https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip',
        'https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip'
    ]
    raw_folder = 'raw'
    word2sents = 'word2sent.txt'
    training_file = 'training.pt'
    test_file = 'test.pt'

    '''
    The items are (filename,category). The index of all the categories can be found in self.idx_classes
    Args:
    - root: the directory where the dataset will be stored
    - transform: how to transform the input
    - target_transform: how to transform the target
    - download: need to download the dataset
    '''

    def __init__(self, root):
        self.root = root

        if not self._check_exists():
            raise RuntimeError('Dataset not found.'
                               + ' You can use download=True to download it')

        self.all_items, self.word2id, self.max_len = find_classes(os.path.join(self.root, self.word2sents))
        self.idx_classes = index_classes(self.all_items)

    def __getitem__(self, index):
        sent = self.all_items[index][0]
        target = self.idx_classes[self.all_items[index][1]]

        return sent, target

    def __len__(self):
        return len(self.all_items)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.word2sents))


def find_classes(pth):
    retour = []
    word2id = {'<padding>': 0}
    max_len = 0
    with open(pth, 'r') as fr:
        line = fr.readline()
        jdata = json.loads(line.strip())
        for k, values in jdata.items():
            for v in values:
                max_len = max(max_len, len(v))
                retour.append((v, k))
                for w in v:
                    if w not in word2id:
                        word2id[w] = len(word2id)

    print("== Found %d items " % len(retour))
    return retour, word2id, max_len


def index_classes(items):
    idx = {}
    for i in items:
        if (not i[1] in idx):
            idx[i[1]] = len(idx)
    print("== Found %d classes" % len(idx))
    return idx
