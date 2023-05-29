# Federated Learning Based on Model Discrepancy and Variance Reduction

This directory contains source code for evaluating federated learning with different optimizers on various models and tasks. The code was developed for a paper, "Federated Learning Based on Model Discrepancy and Variance Reduction".

## Requirements

Some pip packages are required by this library, and may need to be installed. For more details, see `requirements.txt`. We recommend running `pip install --requirement "requirements.txt"`.

Below we give a summary of the datasets, tasks, and models used in this code.


## Task and dataset summary

Note that we put the dataset under the directory .\federated-learning-master\Folder

<!-- mdformat off(This table is sensitive to automatic formatting changes) -->

| Directory        | Model                               | Task Summary              |
|------------------|-------------------------------------|---------------------------|
| CIFAR-10         | CNN (with two convolutional layers) | Image classification      |
| CIFAR-100        | CNN (with two convolutional layers) | Image classification      |
| EMNIST           | NN (fully connected neural network) | Digit recognition         |
| Shakespeare      | RNN with 2 LSTM layers              | Next-character prediction |

<!-- mdformat on -->


## Training
In this code, we compare 5 optimization methods: **FedMDVR**, **FedDyn**, **Scaffold**, **FedAvgM**, and **FedAvg** ( **FedVR** in another code). Those methods use vanilla SGD on clients. To recreate our experimental results for each optimizer, for example, for 100 clients and 10% participation rate, on the cifar100 data set with Dirichlet (0.25) split, run those commands for different methods:

**FedMDVR**:
```
python /federated-learning-master/main_fed.py --seed 200  --epochs 4000 --iid Dirichlet --rule_arg 0.25 --weigh_delay 0.001 --dataset CIFAR100 --model cnn --gpu 0 --num_users 100 --frac 0.1 --lr 0.1 --local_ep 5 --local_bs 50 --beta_ 0 --alpha 9 --lr_decay 0.998 --method FedMDVR --coe 0 --filepath FedMDVR_cifar100_lr0.1_0.998_frac0.1_beta0_alpha9_coe0.0Dirichlet0.25
```

**FedDyn**:
```
python /federated-learning-master/main_fed.py --seed 200  --epochs 4000 --iid Dirichlet --rule_arg 0.25 --weigh_delay 0.001 --dataset CIFAR100 --model cnn --gpu 0 --num_users 100 --frac 0.1 --lr 0.1 --local_ep 5 --local_bs 50 --beta_ 0 --alpha 0 --lr_decay 0.998 --method feddyn --coe 0.1 --filepath feddyn_cifar100_lr0.1_0.998_frac0.1_beta0_alpha0_coe0.1Dirichlet0.25
```

**Scaffold**:
```
python /federated-learning-master/main_fed.py --seed 200  --epochs 4000 --iid Dirichlet --rule_arg 0.25 --weigh_delay 0.001 --dataset CIFAR100 --model cnn --gpu 0 --num_users 100 --frac 0.1 --lr 0.1 --local_ep 5 --local_bs 50 --beta_ 0 --alpha 0 --lr_decay 0.998 --method scaffold --coe 0 --filepath scaffold_cifar100_lr0.1_0.998_frac0.1_beta0_alpha0_coe0Dirichlet0.25
```

**FedAvgM**:
```
python /federated-learning-master/main_fed.py --seed 200  --epochs 4000 --iid Dirichlet --rule_arg 0.25 --weigh_delay 0.001 --dataset CIFAR100 --model cnn --gpu 0 --num_users 100 --frac 0.1 --lr 0.1 --local_ep 5 --local_bs 50 --beta_ 0.9 --alpha 0 --lr_decay 0.998 --method baseline --coe 0.0 --filepath baseline_cifar100_lr0.1_0.998_frac0.1_beta0.9_alpha0_coe0beta0.9Dirichlet0.25
```

**FedAvg**:
```
python /federated-learning-master/main_fed.py --seed 200  --epochs 4000 --iid Dirichlet --rule_arg 0.25 --weigh_delay 0.001 --dataset CIFAR100 --model cnn --gpu 0 --num_users 100 --frac 0.1 --lr 0.1 --local_ep 5 --local_bs 50 --beta_ 0.0 --alpha 0 --lr_decay 0.998 --method baseline --coe 0 --filepath baseline_cifar100_lr0.1_0.998_frac0.1_beta0_alpha0_coe0beta0.0Dirichlet0.25
```

>📋  ```--alpha``` and ```--coe``` are hyperparameters for **FedMDVR** and **FedDyn**, respectively. **FedAvgM** uses server momentum of 0.9 as  ```--beta_ 0.9 ```


## Other hyperparameters and reproducibility

All other hyperparameters are set by default to the values used in the `Experiment Details` of our Appendix. This includes the batch size, the number of clients per round, the number of client local updates, local learning rate, and model parameter flags. While they can be set for different behavior (such as varying the number of client local updates), they should not be changed if one wishes to reproduce the results from our paper. 

