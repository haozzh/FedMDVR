import torch
import copy
import math
from torch import nn, autograd
import numpy as np
from torch.utils.data import DataLoader
from utils.dataset import Dataset
import torch.nn.functional as F
from utils.model_config import RUNCONFIGS


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


def computer_norm(source1, source2):
    diff_norm = 0

    for name in source1:
        diff_source = source1[name].data.clone() - source2[name].data.clone()
        diff_norm += torch.pow(torch.norm(diff_source),2)

    return (torch.pow(diff_norm, 0.5))

def majority_vote(target, sources, lr):
    for name in target:
        threshs = torch.stack([torch.max(source[name].data) for source in sources])
        mask = torch.stack([source[name].data.sign() for source in sources]).sum(dim=0).sign()
        target[name].data = (lr * mask).clone()



class DistributedTrainingDevice(object):
    '''
  A distributed training device (Client or Server)
  data : a pytorch dataset consisting datapoints (x,y)
  model : a pytorch neural net f mapping x -> f(x)=y_
  hyperparameters : a python dict containing all hyperparameters
  '''

    def __init__(self, model, args, dataset_name):
        self.model = model
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.ensemble_loss=nn.KLDivLoss(reduction="batchmean")
        self.unique_labels = RUNCONFIGS[dataset_name]['unique_labels']
        self.generative_alpha = RUNCONFIGS[dataset_name]['generative_alpha']
        self.generative_beta = RUNCONFIGS[dataset_name]['generative_beta']

class Client(DistributedTrainingDevice):

    def __init__(self, model, args, trn_x, trn_y, dataset_name, available_labels, id_num=0):
        super().__init__(model, args, dataset_name)

        self.trn_gen = DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name),
                                  batch_size=self.args.local_bs, shuffle=True)

        #self.available_labels, self.labels_counts = torch.unique(torch.tensor(trn_y).float(), return_counts=True)
        self.id = id_num
        self.local_epoch = int(np.ceil(trn_x.shape[0] / self.args.local_bs))
        # Parameters
        self.W = {name: value for name, value in self.model.named_parameters()}
        self.W_old = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}
        self.dW = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}
        self.state_params_diff = 0.0
        self.train_loss = 0.0

        self.available_labels = available_labels
        self.latent_layer_idx = -1

        self.label_counts = {label:1 for label in range(self.unique_labels)}
        self.labels_aviable, self.labels_counts = torch.unique(torch.tensor(trn_y).float(), return_counts=True)
        for label, count in zip(self.labels_aviable, self.labels_counts):
            self.label_counts[int(label)] += count

    def synchronize_with_server(self, server):
        # W_client = W_server
        self.model = copy.deepcopy(server.model)
        self.W = {name: value for name, value in self.model.named_parameters()}

    def exp_lr_scheduler(self, epoch, decay=0.98, init_lr=0.1, lr_decay_epoch=1):
        """Decay learning rate by a factor of 0.95 every lr_decay_epoch epochs."""
        lr= max(1e-4, init_lr * (decay ** (epoch // lr_decay_epoch)))
        return lr

    def update_label_counts(self, labels, counts):
        for label, count in zip(labels, counts):
            self.label_counts[int(label)] += count




    def train_cnn(self, server, global_iter, early_stop=100, regularization=True):

        self.model.train()
        server.generative_model.eval()


        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.weigh_delay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1)

        # train and update
        epoch_loss = []
        TEACHER_LOSS, DIST_LOSS, LATENT_LOSS = 0, 0, 0
        for iter in range(self.args.local_ep):
            trn_gen_iter = self.trn_gen.__iter__()
            batch_loss = []
            for i in range(self.local_epoch):

                images, labels = trn_gen_iter.__next__()
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                optimizer.zero_grad()
                log_probs = self.model(images)['logit']
                log_probs_out = self.model(images)['output']
                loss_f_i = self.loss_func(log_probs, labels.reshape(-1).long())

                if regularization and iter < early_stop and global_iter > 0:
                    generative_alpha=self.exp_lr_scheduler(iter, decay=0.98, init_lr=self.generative_alpha)
                    generative_beta=self.exp_lr_scheduler(iter, decay=0.98, init_lr=self.generative_beta)
                    gen_output=server.generative_model(labels.reshape(-1).long(), latent_layer_idx=server.latent_layer_idx)['output']
                    logit_given_gen=self.model(gen_output, start_layer_idx=self.latent_layer_idx, logit=True)['logit']
                    target_p=F.softmax(logit_given_gen, dim=1).clone().detach()
                    user_latent_loss= generative_beta * self.ensemble_loss(log_probs_out, target_p)


                    sampled_y=np.random.choice(self.available_labels, self.args.gen_batch_size)
                    sampled_y=torch.tensor(sampled_y).to(self.args.device)
                    gen_result=server.generative_model(sampled_y, latent_layer_idx=self.latent_layer_idx)
                    gen_output=gen_result['output'] # latent representation when latent = True, x otherwise
                    user_output_logp =self.model(gen_output, start_layer_idx=self.latent_layer_idx)['output']
                    sampled_y = torch.tensor(sampled_y, device=self.args.device, dtype=torch.long)
                    teacher_loss =  generative_alpha * torch.mean(
                        server.generative_model.crossentropy_loss(user_output_logp, sampled_y)
                    )
                    # this is to further balance oversampled down-sampled synthetic data
                    gen_ratio = self.args.gen_batch_size / self.args.local_bs
                    loss=loss_f_i + gen_ratio * teacher_loss + user_latent_loss
                    TEACHER_LOSS+=teacher_loss
                    LATENT_LOSS+=user_latent_loss

                else:
                    loss = loss_f_i

                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=max_norm)
                optimizer.step()

                batch_loss.append(loss.item())

            scheduler.step()
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return sum(epoch_loss) / len(epoch_loss)

    def compute_weight_update(self, server, global_iter):


        # Training mode
        self.model.train()

        # W_old = W
        self.W_old = copy.deepcopy(self.W)

        # W = SGD(W, D)
        self.train_loss = self.train_cnn(server, global_iter)

        # dW = W - W_old
        subtract_(target=self.dW, minuend=self.W, subtrahend=self.W_old)




class Server(DistributedTrainingDevice):

    def __init__(self, model, args, generative_model, available_labels, dataset_name):
        super().__init__(model, args, dataset_name)

        # Parameters
        self.W = {name: value for name, value in self.model.named_parameters()}
        self.dW_now = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}
        self.dW = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}
        self.mome = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}
        self.gnag = {name: torch.zeros(value.shape).to(self.args.device) for name, value in self.W.items()}
        self.all_model = copy.deepcopy(model).to(args.device)
        self.all_W = {name: value for name, value in self.all_model.named_parameters()}
        self.local_epoch = 0

        self.available_labels = available_labels
        self.unique_labels = RUNCONFIGS[dataset_name]['unique_labels']
        self.generative_model = generative_model
        self.latent_layer_idx = self.generative_model.latent_layer_idx
        self.ensemble_alpha, self.ensemble_eta = RUNCONFIGS[args.dataset].get('ensemble_alpha', 1), RUNCONFIGS[args.dataset].get('ensemble_eta', 1)
        self.ensemble_lr = RUNCONFIGS[self.args.dataset].get('ensemble_lr', 1e-4)
        self.weight_decay = RUNCONFIGS[self.args.dataset].get('weight_decay', 0)
        self.loss=nn.NLLLoss()
        self.ensemble_loss=nn.KLDivLoss(reduction="batchmean")#,log_target=True)
        self.ce_loss = nn.CrossEntropyLoss()

        self.generative_optimizer = torch.optim.Adam(
            params=self.generative_model.parameters(),
            lr=self.ensemble_lr, betas=(0.9, 0.999),
            eps=1e-08, weight_decay=self.weight_decay, amsgrad=False)

        self.generative_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.generative_optimizer, gamma=0.98)

    def get_label_weights(self, selected_clients):
        MIN_SAMPLES_PER_LABEL = 1
        label_weights = []
        qualified_labels = []
        for label in range(self.unique_labels):
            weights = []
            for user in selected_clients:
                weights.append(user.label_counts[label])
            if np.max(weights) > MIN_SAMPLES_PER_LABEL:
                qualified_labels.append(label)
            # uniform
            #weights = torch.tensor(weights, device = 'cpu')
            label_weights.append(np.array(weights) / np.sum(weights))
        label_weights = np.array(label_weights).reshape((self.unique_labels, -1))
        return label_weights, qualified_labels

    def train_generator(self, selected_clients, latent_layer_idx=-1):

        #self.generative_regularizer.train()
        self.label_weights, self.qualified_labels = self.get_label_weights(selected_clients)
        TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS, STUDENT_LOSS2 = 0, 0, 0, 0
        self.generative_model.train()

        for i in range(self.args.global_ep):
            self.generative_optimizer.zero_grad()
            y=np.random.choice(self.qualified_labels, self.args.server_gen_bs)
            y_input=torch.LongTensor(y).to(self.args.device)
            ## feed to generator
            gen_result=self.generative_model(y_input, latent_layer_idx=latent_layer_idx, verbose=True)
            # get approximation of Z( latent) if latent set to True, X( raw image) otherwise
            gen_output, eps=gen_result['output'], gen_result['eps']
            ##### get losses ####
            # decoded = self.generative_regularizer(gen_output)
            # regularization_loss = beta * self.generative_model.dist_loss(decoded, eps) # map generated z back to eps
            diversity_loss=self.generative_model.diversity_loss(eps, gen_output)  # encourage different outputs

            ######### get teacher loss ############
            teacher_loss=0
            teacher_logit=0

            for user_idx, user in enumerate(selected_clients):
                user.model.eval()
                weight=self.label_weights[y][:, user_idx].reshape(-1, 1)
                expand_weight=np.tile(weight, (1, self.unique_labels))
                user_result_given_gen=user.model(gen_output, start_layer_idx=latent_layer_idx, logit=True)
                user_output_logp_ = F.log_softmax(user_result_given_gen['logit'], dim=1)
                teacher_loss_=torch.mean( \
                    self.generative_model.crossentropy_loss(user_output_logp_, y_input) * \
                    torch.tensor(weight, dtype=torch.float32).to(self.args.device))
                teacher_loss+=teacher_loss_
                teacher_logit+=user_result_given_gen['logit'] * (torch.tensor(expand_weight, dtype=torch.float32).to(self.args.device))

            loss=self.ensemble_alpha * teacher_loss + self.ensemble_eta * diversity_loss

            loss.backward()
            self.generative_optimizer.step()
            TEACHER_LOSS += self.ensemble_alpha * teacher_loss#(torch.mean(TEACHER_LOSS.double())).item()

            DIVERSITY_LOSS += self.ensemble_eta * diversity_loss#(torch.mean(diversity_loss.double())).item()

            self.generative_lr_scheduler.step()
        return TEACHER_LOSS, DIVERSITY_LOSS


    def aggregate_weight_updates(self, clients, iter, aggregation="mean"):

        # Warning: Note that K is different for unbalanced dataset
        self.local_epoch = clients[0].local_epoch
        # dW = aggregate(dW_i, i=1,..,n)
        if aggregation == "mean":
            average(target=self.dW, sources=[client.dW for client in clients])

    def computer_weight_update_down_dw(self, clients, iter):

        # Warning: Note that K is different for unbalanced dataset
        self.local_epoch = clients[0].local_epoch
        add_mome2(target=self.W, source1=self.W, source2=self.dW, beta_1=1, beta_2=self.args.globallr)


    @torch.no_grad()
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
            log_probs = self.model(data)['logit']
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

    @torch.no_grad()
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
            log_probs = self.all_model(data)['logit']
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

