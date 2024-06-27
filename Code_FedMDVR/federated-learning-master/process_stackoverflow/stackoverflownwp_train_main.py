from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from rnn import *
import os
import json
import numpy as np
import random
from torch.utils.data import Dataset
from utils.options import args_parser

#TODO do something about this

class CustomTensorDataset(Dataset):
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


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        test = target.reshape(-1).long()
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()
        # if batch_idx % args.log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / (len(train_loader)), loss.item()))
        #     if args.dry_run:
        #         break

def test(model, device, test_loader, test_data_x):
    model.eval()
    test_loss = 0
    correct = 0
    n_tst = test_data_x.shape[0]
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.CrossEntropyLoss(reduction='sum')(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            #output = output.cpu().detach().numpy()
            #log_probs = np.argmax(output, axis=1).reshape(-1)
            #target = target.cpu().numpy().reshape(-1).astype(np.int32)
            #correct += np.sum(log_probs == target)

    test_loss /= (len(test_loader.dataset) * 20) #20 is the length of the sentence

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, n_tst,
        100. * correct / (20 * n_tst)))

    #print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        #test_loss, correct, len(test_loader.dataset),
        #100. * correct / (20 * (n_tst))))

def process():

    args = args_parser()

    os.environ['PYTHONHASHSEED']=str(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    train_data_x, train_data_y, train_data_dict, test_data_x, test_data_y = torch.load('Stackoverflownwp_data_split.pt')

    train_dataset_aug = CustomTensorDataset(tensors=(train_data_dict[0]["x"], train_data_dict[0]["y"]))
    train_loader = torch.utils.data.DataLoader(train_dataset_aug, batch_size=args.bs, shuffle=True)#, num_workers = 2)

    test_dataset_aug = CustomTensorDataset(tensors=(test_data_x, test_data_y))
    test_loader = torch.utils.data.DataLoader(test_dataset_aug, batch_size=100, shuffle=False)#, num_workers = 2)

    model = RNN_StackOverFlow().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma_)

    for epoch in range(args.epochs):
        train(args, model, device, train_loader, optimizer, epoch)

        if epoch%5==0:
            test(model, device, test_loader, test_data_x)
        scheduler.step()

if __name__ == '__main__':
    process()
