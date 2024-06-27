import torch
import copy
import math
from torch import nn, autograd
import numpy as np
from torch.utils.data import DataLoader
from utils.dataset import Dataset
import torch.nn.functional as F


max_norm = 10


def add(target, source):
    for name in target:
        target[name].data += source[name].data.clone()


def add_mome(target, source, beta_):
    for name in target:
        target[name].data = (beta_ * target[name].data + source[name].data.clone())


def add_mome2(target, source1, source2, beta_1, beta_2):
    for name in target:
        target[name].data = beta_1 * source1[name].data.clone() + beta_2 * source2[name].data.clone()


def add_mome3(target, source1, source2, source3, beta_1, beta_2, beta_3):
    for name in target:
        target[name].data = beta_1 * source1[name].data.clone() + beta_2 * source2[name].data.clone() + beta_3 * \
                            source3[name].data.clone()


def add_2(target, source1, source2, beta_1, beta_2):
    for name in target:
        target[name].data += beta_1 * source1[name].data.clone() + beta_2 * source2[name].data.clone()


def scale(target, scaling):
    for name in target:
        target[name].data = scaling * target[name].data.clone()


def scale_ts(target, source, scaling):
    for name in target:
        target[name].data = scaling * source[name].data.clone()


def subtract(target, source):
    for name in target:
        target[name].data -= source[name].data.clone()


def subtract_(target, minuend, subtrahend):
    for name in target:
        target[name].data = minuend[name].data.clone() - subtrahend[name].data.clone()


def average(target, sources):
    for name in target:
        target[name].data = torch.mean(torch.stack([source[name].data for source in sources]), dim=0).clone()


def weighted_average(target, sources, weights):
    for name in target:
        summ = torch.sum(weights)
        n = len(sources)
        modify = [weight / summ * n for weight in weights]
        target[name].data = torch.mean(torch.stack([m * source[name].data for source, m in zip(sources, modify)]),
                                       dim=0).clone()


def majority_vote(target, sources, lr):
    for name in target:
        threshs = torch.stack([torch.max(source[name].data) for source in sources])
        mask = torch.stack([source[name].data.sign() for source in sources]).sum(dim=0).sign()
        target[name].data = (lr * mask).clone()


def get_mdl_params(model_list, n_par=None):
    if n_par == None:
        exp_mdl = model_list[0]
        n_par = 0
        for name, param in exp_mdl.named_parameters():
            n_par += len(param.data.reshape(-1))

    param_mat = np.zeros((len(model_list), n_par)).astype('float32')
    for i, mdl in enumerate(model_list):
        idx = 0
        for name, param in mdl.named_parameters():
            temp = param.data.cpu().numpy().reshape(-1)
            param_mat[i, idx:idx + len(temp)] = temp
            idx += len(temp)
    return np.copy(param_mat)


def get_other_params(model_list, n_par=None):
    if n_par == None:
        exp_mdl = model_list[0]
        n_par = 0
        for name in exp_mdl:
            n_par += len(exp_mdl[name].data.reshape(-1))

    param_mat = np.zeros((len(model_list), n_par)).astype('float32')
    for i, mdl in enumerate(model_list):
        idx = 0
        for name in mdl:
            temp = mdl[name].data.cpu().numpy().reshape(-1)
            param_mat[i, idx:idx + len(temp)] = temp
            idx += len(temp)
    return np.copy(param_mat)


class DistributedTrainingDevice(object):
    '''
  A distributed training device (Client or Server)
  data : a pytorch dataset consisting datapoints (x,y)
  model : a pytorch neural net f mapping x -> f(x)=y_
  hyperparameters : a python dict containing all hyperparameters
  '''

    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()


class Client(DistributedTrainingDevice):

    def __init__(self, model, args, trn_x, trn_y, dataset_name, id_num=0):
        super().__init__(model, args)

        self.trn_gen = DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name),
                                  batch_size=self.args.local_bs, shuffle=True)

        self.local_epoch = int(np.ceil(trn_x.shape[0] / self.args.local_bs))
        # Parameters
        self.W = {name: value for name, value in self.model.named_parameters()}
        self.W_old = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}
        self.dW = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}
        self.dW_old = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}
        self.dd = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}

        self.train_loss = 0.0

    def synchronize_with_server(self, server):
        # W_client = W_server
        self.model = copy.deepcopy(server.model)
        self.W = {name: value for name, value in self.model.named_parameters()}

    def train_cnn(self, server):

        self.model.train()

        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum,
                                    weight_decay=self.args.weigh_delay)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1)

        # train and update
        epoch_loss = []
        for iter in range(self.args.local_ep):
            trn_gen_iter = self.trn_gen.__iter__()
            batch_loss = []
            for i in range(self.local_epoch):
                images, labels = trn_gen_iter.__next__()
                images, labels = images.to(self.args.device), labels.to(self.args.device)

                optimizer.zero_grad()
                log_probs = self.model(images)
                loss_f_i = self.loss_func(log_probs, labels.reshape(-1).long())

                loss = loss_f_i

                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=max_norm)
                optimizer.step()

                batch_loss.append(loss.item())
            scheduler.step()
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return sum(epoch_loss) / len(epoch_loss)

    def compute_weight_update(self, server):

        # Training mode
        self.model.train()

        # W_old = W
        self.W_old = copy.deepcopy(self.W)

        # W = SGD(W, D)
        self.train_loss = self.train_cnn(server)

        # dW = W - W_old
        subtract_(target=self.dW, minuend=self.W, subtrahend=self.W_old)
        subtract_(target=self.dd, minuend=self.dW, subtrahend=self.dW_old)
        scale_ts(target=self.dW_old, source=self.dW, scaling=1)


class Server(DistributedTrainingDevice):

    def __init__(self, model, args):
        super().__init__(model, args)

        # Parameters
        self.W = {name: value for name, value in self.model.named_parameters()}
        self.dW_now = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}
        self.h = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}
        self.dd = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}


    def aggregate_weight_updates(self, clients, iter, aggregation="mean"):

        # Warning: Note that K is different for unbalanced dataset
        # dW = aggregate(dW_i, i=1,..,n)

        if self.args.method == 'sag':
            average(target=self.dd, sources=[client.dd for client in clients])

            add_mome2(target=self.h, source1=self.h, source2=self.dd, beta_1=1,
                      beta_2=self.args.num_users * self.args.frac)
            scale_ts(target=self.dW_now, source=self.h, scaling=1 / self.args.num_users)

        elif self.args.method == 'saga':
            average(target=self.dd, sources=[client.dd for client in clients])

            add_mome2(target=self.dW_now, source1=self.h, source2=self.dd, beta_1=self.args.alpha / self.args.num_users,
                      beta_2=1)
            add_mome2(target=self.h, source1=self.h, source2=self.dd, beta_1=1,
                      beta_2=self.args.num_users * self.args.frac)

        else:
            exit('Error: unrecognized parameter')

        scale(target=self.dW_now, scaling=self.args.globallr)

        add(target=self.W, source=self.dW_now)  # FedAvg

    def evaluate(self, data_x, data_y, dataset_name):
        self.model.eval()
        # testing
        test_loss = 0
        acc_overall = 0
        n_tst = data_x.shape[0]
        tst_gen = DataLoader(Dataset(data_x, data_y, dataset_name=dataset_name), batch_size=self.args.bs, shuffle=False)
        tst_gen_iter = tst_gen.__iter__()
        for i in range(int(np.ceil(n_tst / self.args.bs))):
            data, target = tst_gen_iter.__next__()
            data, target = data.to(self.args.device), target.to(self.args.device)
            log_probs = self.model(data)
            # sum up batch loss
            test_loss += nn.CrossEntropyLoss(reduction='sum')(log_probs, target.reshape(-1).long()).item()
            # get the index of the max log-probability
            log_probs = log_probs.cpu().detach().numpy()
            log_probs = np.argmax(log_probs, axis=1).reshape(-1)
            target = target.cpu().numpy().reshape(-1).astype(np.int32)
            batch_correct = np.sum(log_probs == target)
            acc_overall += batch_correct
            '''
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
            '''
        test_loss /= n_tst
        accuracy = 100.00 * acc_overall / n_tst
        return accuracy, test_loss

