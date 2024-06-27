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
        target[name].data = beta_1 * source1[name].data.clone() + beta_2 * source2[name].data.clone() + beta_3 * source3[name].data.clone()

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


def get_model_norm(target):
    model_norm = torch.norm(torch.cat([target[name].data.view(-1) for name in target], dim = 0)).clone()
    return model_norm


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
        self.id = id_num
        self.local_epoch = int(np.ceil(trn_x.shape[0] / self.args.local_bs))
        self.number_data = trn_x.shape[0]
        # Parameters
        self.W = {name: value for name, value in self.model.named_parameters()}
        self.W_old = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}
        self.dW = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}
        self.dg = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}
        self.mome = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}
        self.unbias = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}

        if self.args.method == 'scaffold':
            self.ci = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}
            self.delta_ci = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}
            self.c_plus = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}

        if self.args.method == 'feddyn':
            self.histi = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}

        self.state_params_diff = 0.0
        self.train_loss = 0.0
        self.n_par = get_mdl_params([self.model]).shape[1]

    def synchronize_with_server(self, server):
        # W_client = W_server
        self.model = copy.deepcopy(server.model)
        self.W = {name: value for name, value in self.model.named_parameters()}


    def gaussian_noise(self, data_):
        # works with integers as well as tuples
        Gaussian_ = {name: torch.randn_like(value).to(self.args.device) for name, value in data_.items()}
        return Gaussian_


    def compute_bias(self, server):

        if self.args.method == 'FedMDVR':
            for name in self.unbias:
                self.unbias[name].data += (-server.h[name].data.clone()) + self.dg[name].data.clone()
            self.state_params_diff = torch.tensor(get_other_params([self.unbias], self.n_par)[0], dtype=torch.float32,
                                                  device=self.args.device)

        if self.args.method == 'scaffold':
            for name in self.unbias:
                self.unbias[name].data = server.c[name].data.clone() - self.ci[name].data.clone()
            self.state_params_diff = torch.tensor(get_other_params([self.unbias], self.n_par)[0], dtype=torch.float32,
                                                  device=self.args.device)

        if self.args.method == 'feddyn':
            cld_mdl_param = torch.tensor(get_mdl_params([self.model], self.n_par)[0], dtype=torch.float32, device=self.args.device)
            hist_mdl_param = torch.tensor(get_other_params([self.histi], self.n_par)[0], dtype=torch.float32, device=self.args.device)
            self.state_params_diff = self.args.coe * (-cld_mdl_param + hist_mdl_param)

    def train_cnn(self, server):

        self.model.train()

        if self.args.method == 'FedMDVR' or self.args.method == 'scaffold' or self.args.method == 'baseline':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum,
                                        weight_decay=self.args.weigh_delay)
        if self.args.method == 'feddyn':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum,
                                        weight_decay=self.args.weigh_delay + self.args.coe)

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

                local_par_list = None
                for param in self.model.parameters():
                    if not isinstance(local_par_list, torch.Tensor):
                        # Initially nothing to concatenate
                        local_par_list = param.reshape(-1)
                    else:
                        local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)

                loss_algo = torch.sum(local_par_list * self.state_params_diff)
                loss = loss_f_i + loss_algo

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

        scale_ts(target=self.dg, source=self.dW,
                 scaling=(1 / (self.args.lr * self.local_epoch * self.args.local_ep)))

        if self.args.method == 'scaffold':
            add_mome3(target=self.c_plus, source1=self.ci, source2=server.c, source3=self.dg, beta_1=1,
                      beta_2=-1, beta_3=-1)
            add_mome2(target=self.delta_ci, source1=self.c_plus, source2=self.ci, beta_1=1, beta_2=-1)
            scale_ts(target=self.ci, source=self.c_plus, scaling=1)

        if self.args.method == 'feddyn':
            add(target=self.histi, source=self.dW)

    def add_upload(self):
        gaussian_sample = self.gaussian_noise(self.W_old)
        cc = math.pow(2 * math.log(1.25 / self.args.dp_delta),0.5)
        C_clip = get_model_norm(self.W)
        s_u =2 * C_clip / self.number_data
        sigma_u = cc *  self.args.L_exposures * s_u / self.args.dp_epision
        add_mome2(target=self.dW, source1=self.dW, source2=gaussian_sample, beta_1 = 1, beta_2 = math.pow(sigma_u, 2))


class Server(DistributedTrainingDevice):

    def __init__(self, model, args):
        super().__init__(model, args)

        # Parameters
        self.W = {name: value for name, value in self.model.named_parameters()}
        self.dW_now = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}
        self.dW = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}
        self.mome = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}
        self.all_model = copy.deepcopy(model).to(args.device)
        self.all_W = {name: value for name, value in self.all_model.named_parameters()}
        if self.args.method == 'FedMDVR':
            self.h = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}

        if self.args.method == 'scaffold':
            self.c = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}
            self.delta_c = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}

        if self.args.method == 'feddyn':
            self.hist = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}
            self.delta_dyn= {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}

        self.local_epoch = 0

    def gaussian_noise(self, data_):
        # works with integers as well as tuples
        Gaussian_ = {name: torch.randn_like(value).to(self.args.device) for name, value in data_.items()}
        return Gaussian_



    def aggregate_weight_updates(self, clients, iter, aggregation="mean"):

        # Warning: Note that K is different for unbalanced dataset
        self.local_epoch = clients[0].local_epoch
        # dW = aggregate(dW_i, i=1,..,n)
        if aggregation == "mean":
            average(target=self.dW, sources=[client.dW for client in clients])


        elif aggregation == "weighted_mean":
            weighted_average(target=self.dW, sources=[client.dW for client in clients],
                             weights=torch.stack([self.client_sizes[client.id] for client in clients]))

        elif aggregation == "majority":
            majority_vote(target=self.dW, sources=[client.dW for client in clients], lr=self.hp["lr"])

        add_mome(target=self.mome, source=self.dW, beta_=self.args.beta_) #FedAvgM

    def computer_weight_update_down_dw(self, clients, iter):

        # Warning: Note that K is different for unbalanced dataset
        self.local_epoch = clients[0].local_epoch

        if self.args.method == 'scaffold':
            add(target=self.W, source=self.mome)

            average(target=self.delta_c, sources=[client.delta_ci for client in clients])
            scale(target=self.delta_c, scaling=self.args.frac)
            add(target=self.c, source=self.delta_c)

        if self.args.method == 'FedMDVR':
            add(target=self.W, source=self.mome)

            scale_ts(target=self.dW_now, source=self.mome, scaling=(1 - self.args.beta_) / (1 - math.pow(self.args.beta_, iter + 1)))
            scale(target=self.dW_now, scaling=(1 / (self.args.lr * self.local_epoch * self.args.local_ep)))

            add_mome2(target=self.h, source1=self.h, source2=self.dW_now, beta_1=(1-self.args.alpha*self.args.frac), beta_2=self.args.alpha*self.args.frac)

        if self.args.method == 'feddyn':
            add(target=self.W, source=self.mome)

            scale_ts(target=self.delta_dyn, source=self.mome, scaling=self.args.frac)
            add(target=self.hist, source=self.delta_dyn)
            add(target=self.W, source=self.hist)

        if self.args.method == 'baseline':
            add(target=self.W, source=self.mome) #FedAvg


    def add_download(self, iter, m, number_data):
        gaussian_sample = self.gaussian_noise(self.dW)
        cc = math.pow(2 * math.log(1.25 / self.args.dp_delta),0.5)
        C_clip = get_model_norm(self.W)
        s_u =2 * C_clip / number_data
        gamma_ = - math.log(1 - self.args.frac + self.args.frac * math.exp(-self.args.dp_epision / ( self.args.L_exposures * math.sqrt(m))))

        if self.args.epochs > self.args.dp_epision / gamma_:
            b = -self.args.epochs / self.args.dp_epision * math.log(
                1 - 1 / self.args.frac + (1 / self.args.frac) * math.exp(-self.args.dp_epision / self.args.epochs))
            sigma_u = cc *  math.sqrt(math.pow((self.args.epochs/b), 2) - math.pow(self.args.L_exposures ,2) * m) * s_u / self.args.dp_epision / m
            add_mome2(target=self.W, source1=self.W, source2=gaussian_sample, beta_1 = 1, beta_2 = math.pow(sigma_u, 2))


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

    def all_model_evaluate(self, clients, data_x, data_y, dataset_name):

        average(target=self.all_W, sources=[client.W for client in clients])
        self.all_model.eval()
        # testing
        test_loss = 0
        acc_overall = 0
        n_tst = data_x.shape[0]
        tst_gen = DataLoader(Dataset(data_x, data_y, dataset_name=dataset_name), batch_size=self.args.bs, shuffle=False)
        tst_gen_iter = tst_gen.__iter__()
        for i in range(int(np.ceil(n_tst / self.args.bs))):
            data, target = tst_gen_iter.__next__()
            data, target = data.to(self.args.device), target.to(self.args.device)
            log_probs = self.all_model(data)
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

