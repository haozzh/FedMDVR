#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=4000, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.05, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=1, help="the number of local epochs: E")
    parser.add_argument('--IASG_epoch', type=int, default=1, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=50, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=50, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.1, help="learning rate")
    parser.add_argument('--globallr', type=float, default=1, help="Global learning rate")
    parser.add_argument('--momentum', type=float, default=0, help="local SGD momentum (default: 0.0)")
    parser.add_argument('--weigh_delay', type=float, default=0, help="local SGD weigh_delay")
    parser.add_argument('--alpha', type=float, default=0, help='the value of alpha')
    parser.add_argument('--beta_', type=float, default=0, help='the value beta_')
    parser.add_argument('--beta_2', type=float, default=0, help='the value of beta_2')
    parser.add_argument('--tau', type=float, default=0, help='the value of fedyogi amd fedadam')
    parser.add_argument('--tau_nova', type=float, default=0, help='the value of fednova')
    parser.add_argument('--gamma_', type=float, default=0, help='the value of fed_localnesterov amd fednesterov')#
    parser.add_argument('--mu', type=float, default=0, help='the value of FedProx')
    parser.add_argument('--coe', type=float, default=0.1, help='the value of Feddyn')
    parser.add_argument('--lr_decay', type=float, default=0.999, help='the value of lr_decay')
    parser.add_argument('--rule_arg', type=float, default=0.25, help='the value of rule_arg')
    parser.add_argument('--iid_scale', type=float, default=1, help='the value of iid_scale')
    parser.add_argument('--local_svrg', type=int, default=0, help='whether local_svrg or not')

    parser.add_argument('--coe_fedgamma', type=float, default=0, help='the value of fedgamma')

    parser.add_argument('--dp_p', type=int, default=0, help='the value of DP')
    parser.add_argument('--dp_delta', type=float, default=0.3, help='the value of DP')
    parser.add_argument('--L_exposures', type=float, default=0.3, help='the value of exposures')
    parser.add_argument('--dp_epision', type=float, default=0.3, help='the value of exposures')

    parser.add_argument('--adacoe', type=float, default=0, help='the value of adabest')
    parser.add_argument('--ada_beta', type=float, default=0, help='the value of adabest')


    parser.add_argument('--filepath', type=str, default='filepath', help='whether error accumulation or not')

    # model arguments
    parser.add_argument('--method', type=str, default='fedciw', help='method name')
    parser.add_argument('--model', type=str, default='cnn', help='model name')

    # other arguments
    parser.add_argument('--dataset', type=str, default='CIFAR100', help="name of dataset")
    parser.add_argument('--iid', type=str, default='Dirichlet', help='whether i.i.d or not')
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--seed', type=int, default=200, help='random seed (default: 23)')
    args = parser.parse_args()
    return args

