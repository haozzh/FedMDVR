from scipy import io
from LEAF.utils_eval.language_utils import letter_to_vec, word_to_indices
from LEAF.utils_eval.model_utils import read_data
import numpy as np
import torchvision
from torchvision import transforms
import torch
from torch.utils import data
import os


class stackoverflowDatasetObject:
    def __init__(self, data_path):

        self.dataset = 'stackoverflow'

        self.train_data_x, self.train_data_y, self.train_data_dict, self.test_data_x, self.test_data_y = torch.load(data_path + 'Stackoverflownwp_data_split.pt')

        self.num_users = len(self.train_data_dict)



class Dataset_stackoverflow(torch.utils.data.Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)