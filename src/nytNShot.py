from src import nytl
from PIL import Image
import os.path

import numpy as np

np.random.seed(2191)  # for reproducibility
filenameToPILImage = lambda x: Image.open(x).convert('L')
PiLImageResize = lambda x: x.resize((28, 28))
np_reshape = lambda x: np.reshape(x, (28, 28, 1))


class nytNShotDataset():
    def __init__(self, dataroot, classes_per_set=10, samples_per_class=1, n_test_samples=1):

        if not os.path.isfile(os.path.join(dataroot, 'data.npy')):
            self.x = nytl.NYT(dataroot)
            self.word2id = self.x.word2id
            self.word_num = len(self.word2id)
            self.max_len = self.x.max_len
            temp = dict()
            for (sent, label) in self.x:
                sent = [self.word2id[w] for w in sent] + [0] * (self.max_len - len(sent))

                if label in temp:
                    temp[label].append(sent)
                else:
                    temp[label] = [sent]
            self.x = []  # Free memory

            for classes in temp.keys():
                self.x.append(np.array(temp[classes]))
            self.x = np.array(self.x)
            temp = []  # Free memory
            np.save(os.path.join(dataroot, 'data.npy'), self.x)
            np.save(os.path.join(dataroot, 'word.npy'), [len(self.word2id), self.max_len])
        else:
            self.x = np.load(os.path.join(dataroot, 'data.npy'))
            wordpara = np.load(os.path.join(dataroot, 'word.npy'))
            self.word_num = wordpara[0]
            self.max_len = wordpara[1]

        """
        Constructs an N-Shot  Dataset
        """
        shuffle_classes = np.arange(self.x.shape[0])
        np.random.shuffle(shuffle_classes)
        self.x = self.x[shuffle_classes]
        self.x_train, self.x_test, self.x_val = self.x[:800], self.x[800:900], self.x[900:]

        self.n_classes = self.x.shape[0]
        self.classes_per_set = classes_per_set
        self.samples_per_class = samples_per_class
        self.n_test_samples = n_test_samples

        self.indexes = {"train": 0, "val": 0, "test": 0}
        self.datasets = {"train": self.x_train, "val": self.x_val, "test": self.x_test}  # original data cached
        self.datasets_cache = {"train": self.load_data_cache(self.datasets["train"]),
                               "val": self.load_data_cache(self.datasets["val"]),
                               "test": self.load_data_cache(self.datasets["test"])}

    def load_data_cache(self, data_pack):
        """
        Collects batches data for N-shot learning
        :param data_pack: Data pack to use (any one of train, val, test)
        :return: A list with [support_set_x, support_set_y, target_x, target_y] ready to be fed to our networks
        """
        n_samples = self.samples_per_class * self.classes_per_set
        test_samples = self.n_test_samples * self.classes_per_set
        data_cache = []
        for sample in range(1000):
            support_set_x = np.zeros((n_samples, self.max_len), dtype=np.int)
            support_set_y = np.zeros((n_samples))
            target_x = np.zeros((test_samples, self.max_len), dtype=np.int)
            target_y = np.zeros((test_samples), dtype=np.int)

            pinds = np.random.permutation(n_samples)
            classes = np.random.choice(data_pack.shape[0], self.classes_per_set, False)
            pinds_test = np.random.permutation(test_samples)
            ind = 0
            ind_test = 0
            for j, cur_class in enumerate(classes):  # each class
                example_inds = np.random.choice(data_pack.shape[1], self.samples_per_class + self.n_test_samples, False)
                # meta-training
                for eind in example_inds[:self.samples_per_class]:
                    support_set_x[pinds[ind], :] = data_pack[cur_class][eind]
                    support_set_y[pinds[ind]] = j
                    ind = ind + 1
                # meta-test
                for eind in example_inds[self.samples_per_class:]:
                    target_x[pinds_test[ind_test], :] = data_pack[cur_class][eind]
                    target_y[pinds_test[ind_test]] = j
                    ind_test = ind_test + 1

            data_cache.append([support_set_x, support_set_y, target_x, target_y])
        return data_cache

    def __get_batch(self, dataset_name):
        """
        Gets next batch from the dataset with name.
        :param dataset_name: The name of the dataset (one of "train", "val", "test")
        :return:
        """
        if self.indexes[dataset_name] >= len(self.datasets_cache[dataset_name]):
            self.indexes[dataset_name] = 0
            self.datasets_cache[dataset_name] = self.load_data_cache(self.datasets[dataset_name])
        next_batch = self.datasets_cache[dataset_name][self.indexes[dataset_name]]
        self.indexes[dataset_name] += 1
        x_support_set, y_support_set, x_target, y_target = next_batch
        return x_support_set, y_support_set, x_target, y_target

    def get_batch(self, str_type):

        """
        Get next batch
        :return: Next batch
        """
        x_support_set, y_support_set, x_target, y_target = self.__get_batch(str_type)

        return x_support_set, y_support_set, x_target, y_target
