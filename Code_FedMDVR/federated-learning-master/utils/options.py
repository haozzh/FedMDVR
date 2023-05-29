#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=10000, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.02, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=50, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=100, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.1, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.0, help="local SGD momentum (default: 0.0)")
    parser.add_argument('--weigh_delay', type=float, default=0.001, help="local SGD weigh_delay")
    parser.add_argument('--alpha', type=float, default=0, help='the value of FEDMDVR')
    parser.add_argument('--beta_', type=float, default=0, help='the value of lambda about global momentum_m (FedAvgM)')
    parser.add_argument('--coe', type=float, default=0.1, help='the value of Feddyn')
    parser.add_argument('--lr_decay', type=float, default=0.999, help='the value of lr_decay')
    parser.add_argument('--rule_arg', type=float, default=0.3, help='the value of rule_arg')

    parser.add_argument('--filepath', type=str, default='compress0', help='whether error accumulation or not')

    # model arguments
    parser.add_argument('--method', type=str, default='FedMDVR', help='model name')
    parser.add_argument('--model', type=str, default='cnn', help='model name')

    # other arguments
    parser.add_argument('--dataset', type=str, default='CIFAR10', help="name of dataset")
    parser.add_argument('--iid', type=str, default='noniid', help='whether i.i.d or not')
    parser.add_argument('--gpu', type=int, default=-1, help="GPU ID, -1 for CPU")
    parser.add_argument('--seed', type=int, default=200, help='random seed (default: 23)')
    args = parser.parse_args()
    return args
