

# temporary roundabout to evaluate sensitivity of the generator
GENERATORCONFIGS = {
    # hidden_dimension, latent_dimension, input_channel, n_class, noise_dim
    'emnist': (256, 100, 1, 10, 32),
    'CIFAR10': (512, 192, 3, 10, 64),
    'CIFAR100': (512, 192, 3, 100, 64),
}



RUNCONFIGS = {
    'emnist':
        {
            'ensemble_lr': 1e-4,
            'ensemble_batch_size': 128,
            'ensemble_epochs': 50,
            'num_pretrain_iters': 20,
            'ensemble_alpha': 1,  # teacher loss (server side)
            'ensemble_beta': 0, # adversarial student loss
            'unique_labels': 10,
            'generative_alpha':10,
            'generative_beta': 1,
            'weight_decay': 1e-2
        },

    'CIFAR10':
        {
            'ensemble_lr': 3e-4,
            'ensemble_batch_size': 128,
            'ensemble_epochs': 50,
            'num_pretrain_iters': 20,
            'ensemble_alpha': 1,    # teacher loss (server side)
            'ensemble_beta': 0,     # adversarial student loss
            'ensemble_eta': 1,      # diversity loss
            'unique_labels': 10,    # available labels
            'generative_alpha': 10, # used to regulate user training
            'generative_beta': 10, # used to regulate user training
            'weight_decay': 1e-2
        },

    'CIFAR100':
        {
            'ensemble_lr': 3e-4,
            'ensemble_batch_size': 128,
            'ensemble_epochs': 50,
            'num_pretrain_iters': 20,
            'ensemble_alpha': 1,  # teacher loss (server side)
            'ensemble_beta': 0,  # adversarial student loss
            'unique_labels': 100,
            'generative_alpha': 10,
            'generative_beta': 10, 
            'weight_decay': 1e-2
        },

}

