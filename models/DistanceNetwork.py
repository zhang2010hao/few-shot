import torch
import torch.nn as nn


class DistanceNetwork(nn.Module):
    def __init__(self):
        super(DistanceNetwork, self).__init__()

    def forward(self, support_set, input_image):
        """
        Produces pdfs over the support set classes for the target set image.
        :param support_set: The embeddings of the support set images, tensor of shape [sequence_length, batch_size, 64]
        :param input_image: The embedding of the target image, tensor of shape [batch_size, 64]
        :return: Softmax pdf. Tensor with cosine similarities of shape [batch_size, sequence_length]
        """
        eps = 1e-10
        input_n = input_image.size(0)
        support_n = support_set.size(0)

        x1 = support_set.unsqueeze(1).expand(-1, input_n, -1)
        x2 = input_image.unsqueeze(0).expand(support_n, -1, -1)
        w12 = torch.sum(x1 * x2, 2)
        w1 = torch.norm(x1, 2, 2)
        w2 = torch.norm(x2, 2, 2)

        cosine_similarity = w12 / (w1 * w2).clamp(min=eps)
        return cosine_similarity
