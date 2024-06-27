from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import os
import json
import numpy as np
import pickle
from rnn import *

from tqdm.notebook import tqdm

from data_loader_after_pretrain_before_preprocess import load_partition_data_federated_stackoverflow_nwp

tup = load_partition_data_federated_stackoverflow_nwp(None, "./datasets/datasets")

dataset = "Stackoverflownwp"

count = 0
a = list(tup[5].keys()).copy()
for i in a:
    if tup[5][i] > 8000:
        del tup[5][i]
        del tup[6][i]
        del tup[7][i]
        count += 1
print(count)



num_users_train = 500  ## custom
num_users_test = 500  ## custom

keys = list(tup[5].keys()).copy()

train_data_dict = dict()
test_data_dict = dict()

glob_train_data_x = []
glob_train_data_y = []
glob_test_data_x = []
glob_test_data_y = []


for i in tqdm(np.arange(num_users_train) + 100):
    user_id = keys[i]
    print(user_id)
    train_data_dict[i-100] = {"x": None, "y": None}
    dum_list_x_train = []
    dum_list_y_train = []

    train_data_user = len(tup[6][user_id].dataset)

    for j in range(train_data_user):
        dum_list_x_train.append(tup[6][user_id].dataset[j][0])
        dum_list_y_train.append(tup[6][user_id].dataset[j][1])

    dum_list_x_train = torch.Tensor(np.stack(dum_list_x_train))
    dum_list_y_train = torch.Tensor(np.stack(dum_list_y_train))
    train_data_dict[i-100]["x"] = dum_list_x_train.long()
    train_data_dict[i-100]["y"] = dum_list_y_train.long()
    glob_train_data_x.append(dum_list_x_train)
    glob_train_data_y.append(dum_list_y_train)

for i in tqdm(np.arange(num_users_test)+100):
    user_id = keys[i]
    print(user_id)
    test_data_dict[i] = {"x": None, "y": None}
    dum_list_x_test = []
    dum_list_y_test = []

    test_data_user = len(tup[7][user_id].dataset)
    indices = np.arange(test_data_user)

    for j in range(test_data_user):
        dum_list_x_test.append(tup[7][user_id].dataset[j][0])
        dum_list_y_test.append(tup[7][user_id].dataset[j][1])

    dum_list_x_test = torch.Tensor(np.stack(dum_list_x_test))
    dum_list_y_test = torch.Tensor(np.stack(dum_list_y_test))
    test_data_dict[i]["x"] = dum_list_x_test.long()
    test_data_dict[i]["y"] = dum_list_y_test.long()
    glob_test_data_x.append(dum_list_x_test)
    glob_test_data_y.append(dum_list_y_test)


glob_train_data_x = torch.cat(glob_train_data_x,dim=0).long()
glob_train_data_y = torch.cat(glob_train_data_y,dim=0).long()

glob_test_data_x = torch.cat(glob_test_data_x,dim=0).long()
glob_test_data_y = torch.cat(glob_test_data_y,dim=0).long()

print(glob_train_data_x.shape)
print(glob_test_data_x.shape)

torch.save((glob_train_data_x, glob_train_data_y, train_data_dict, glob_test_data_x, glob_test_data_y), 'Stackoverflownwp_data_split.pt')
