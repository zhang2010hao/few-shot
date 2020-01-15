import torch
import torch.nn as nn
import numpy as np
from models.BidirectionalLSTM import BidirectionalLSTM
from models.DistanceNetwork import DistanceNetwork
from models.AttentionalClassify import AttentionalClassify
import torch.nn.functional as F
import math
import torch.nn.init as init


def convLayer(in_planes, out_planes, useDropout=False):
    "3x3 convolution with padding"
    seq = nn.Sequential(
        nn.Conv1d(in_planes, out_planes, kernel_size=3,
                  stride=1, padding=1, bias=True),
        nn.BatchNorm1d(out_planes),
        nn.ReLU(True),
        nn.MaxPool1d(kernel_size=2, stride=2)
    )
    if useDropout:  # Add dropout module
        list_seq = list(seq.modules())[1:]
        list_seq.append(nn.Dropout(0.1))
        seq = nn.Sequential(*list_seq)

    return seq


class Classifier(nn.Module):
    def __init__(self, layer_size, nClasses=0, num_channels=1, useDropout=False, image_size=28):
        super(Classifier, self).__init__()

        """
        Builds a CNN to produce embeddings
        :param layer_sizes: A list of length 4 containing the layer sizes
        :param nClasses: If nClasses>0, we want a FC layer at the end with nClasses size.
        :param num_channels: Number of channels of images
        :param useDroput: use Dropout with p=0.1 in each Conv block
        """
        self.layer1 = convLayer(num_channels, layer_size, useDropout)
        self.layer2 = convLayer(layer_size, layer_size, useDropout)
        self.layer3 = convLayer(layer_size, layer_size, useDropout)
        self.layer4 = convLayer(layer_size, layer_size, useDropout)

        finalSize = int(math.floor(image_size / (2 * 2 * 2 * 2)))
        self.outSize = finalSize * layer_size
        if nClasses > 0:  # We want a linear
            self.useClassification = True
            self.layer5 = nn.Linear(self.outSize, nClasses)
            self.outSize = nClasses
        else:
            self.useClassification = False

        self.weights_init(self.layer1)
        self.weights_init(self.layer2)
        self.weights_init(self.layer3)
        self.weights_init(self.layer4)

    def weights_init(self, module):
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform(m.weight, gain=np.sqrt(2))
                init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, image_input):
        """
        Runs the CNN producing the embeddings and the gradients.
        :param image_input: Image input to produce embeddings for. [batch_size, 28, 28, 1]
        :return: Embeddings of size [batch_size, 64]
        """
        x = self.layer1(image_input)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        if self.useClassification:
            x = self.layer5(x)
        return x


class MatchingNetwork(nn.Module):
    def __init__(self, keep_prob, n_word, word_dim=128, num_channels=1, learning_rate=0.001, fce=False,
                 num_classes_per_set=5, num_samples_per_class=1, nClasses=0, sent_len=28):
        super(MatchingNetwork, self).__init__()

        """
        Builds a matching network, the training and evaluation ops as well as data augmentation routines.
        :param keep_prob: A tf placeholder of type tf.float32 denotes the amount of dropout to be used
        :param num_channels: Number of channels of the images
        :param is_training: Flag indicating whether we are training or evaluating
        :param fce: Flag indicating whether to use full context embeddings (i.e. apply an LSTM on the CNN embeddings)
        :param num_classes_per_set: Integer indicating the number of classes per set
        :param num_samples_per_class: Integer indicating the number of samples per class
        :param nClasses: total number of classes. It changes the output size of the classifier g with a final FC layer.
        :param sent_len: max length of the input sentences.  
        """
        self.fce = fce
        self.g = Classifier(layer_size=64, num_channels=num_channels,
                            nClasses=nClasses, image_size=sent_len)
        self.word_embedding = nn.Embedding(n_word,
                                           word_dim)
        if fce:
            self.lstm = BidirectionalLSTM(layer_sizes=[32], vector_dim=self.g.outSize)
        self.dn = DistanceNetwork()
        self.classify = AttentionalClassify()
        self.keep_prob = keep_prob
        self.num_classes_per_set = num_classes_per_set
        self.num_samples_per_class = num_samples_per_class
        self.learning_rate = learning_rate

    def forward(self, support_set_ids, support_set_labels_one_hot, target_ids, target_label):
        """
        Builds graph for Matching Networks, produces losses and summary statistics.
        :param support_set_images: A tensor containing the support set images [batch_size, sequence_size, n_channels, 28, 28]
        :param support_set_labels_one_hot: A tensor containing the support set labels [batch_size, sequence_size, n_classes]
        :param target_image: A tensor containing the target image (image to produce label for) [batch_size, n_channels, 28, 28]
        :param target_label: A tensor containing the target label [batch_size, 1]
        :return:
        """
        # produce embeddings for support set images
        support_set_emb = self.word_embedding(support_set_ids)
        target_emb = self.word_embedding(target_ids)

        encodeds = []

        support_gen_encode = self.g(support_set_emb.permute(0, 2, 1))
        encodeds.append(support_gen_encode)

        target_gen_encode = self.g(target_emb.permute(0, 2, 1))
        similarities = self.dn(support_set=support_gen_encode, input_image=target_gen_encode)
        similarities = similarities.t()
        # produce predictions for target probabilities
        preds = self.classify(similarities, support_set_y=support_set_labels_one_hot)

        # calculate accuracy and crossentropy loss
        values, indices = preds.max(1)

        accuracy = torch.mean((indices.squeeze() == target_label[:]).float())
        crossentropy_loss = F.cross_entropy(preds, target_label[:].long())

        return accuracy, crossentropy_loss
