# Deconstructing Lottery Tickets: Zeros, Signs, and the Supermask

## Authors
Hattie Zhou, Janice Lan, Rosanne Liu, Jason Yosinski

## Introduction
This codebase implements the experiments in [Deconstructing Lottery Tickets: Zeros, Signs, and the Supermask](https://arxiv.org/abs/1905.01067). This paper performs various ablation studies to shine light into the Lottery Tickets (LT) phenomenon observed by Frankle & Carbin in [The Lottery Ticket Hypothesis: Finding Small, Trainable Neural Networks](https://arxiv.org/abs/1803.03635).

```
@inproceedings{dtl
  title={Deconstructing Lottery Tickets: Zeros, Signs, and the Supermask},
  author={Zhou, Hattie and Lan, Janice and Liu, Rosanne and Yosinski, Jason},
  booktitle={Advances in Neural Information Processing Systems},
  year={2019}
}
```

For more on this project, see the [Uber Eng Blog post](https://eng.uber.com/deconstructing-lottery-tickets/).


## Codebase structure
- `data/download_mnist.py`, `data/download_cifar10.py` downloads MNIST/CIFAR10 data and splits it into train, val, and test, and saves them in the `data` folder as `h5` files
- `get_weight_init.py` computes various mask criteria
- `masked_layers.py` defines new layer classes with masking options
- `masked_networks.py` defines new layers and networks used in training Supermasks
- `network_builders.py` defines the four network architecture evaluated in the paper (FC, Conv2, Conv4, Conv6)
- `train.py` trains original unmasked networks
- `train_lottery.py` reads in initial and final weights from a previously trained model, calculates the mask, and train a lottery style network 
- `train_supermask` trains a supermask directly using Bernoulli sampling 
- `get_init_loss_train_lottery.py` derives masks and calculates the initial accuracy of the masked network for various pruning percentages and mask criteria. Note that this uses a one-shot approach rather than an iterative approach.

This codebase uses the `GitResultsManager` package to keep track of experiments. See: https://github.com/yosinski/GitResultsManager

## Example commands for running experiments
The following commands provide examples for running experiments in Deconstructing Lottery Tickets.

#### Train the original, unpruned network
- Train a FC network (300-100-10) on MNIST: `./print_train_command.sh iter fc test 0 t`

#### Alternative mask criteria experiments (using FC on MNIST and large final as an example)
- Perform iterative LT training for a FC network on MNIST using large final mask criterion: `./print_train_lottery_iterative_command.sh fc test 0 large_final -1 mask none t`

#### Mask-1 experiments 
- Randomly reinitialize weights prior to each round of iterative retraining: `./print_train_lottery_iterative_command.sh fc test 0 large_final -1 mask random_reinit t`

- Randomly reshuffle the initial values of remaining weights prior to each round of iterative retraining: `./print_train_lottery_iterative_command.sh fc test 0 large_final -1 mask random_reshuffle t`

- Convert the initial values of weights to a signed constant before randomly reshuffle the initial values of remaining weights prior to each round of iterative retraining: `./print_train_lottery_iterative_command.sh fc test 0 large_final -1 mask rand_signed_constant t`

- For versions that maintain the same sign, see `signed_reinit`, `signed_reshuffle`, and `signed_constant`.

#### Mask-0 experiments
- Freeze pruned weights at initial values: `./print_train_lottery_iterative_command.sh fc test 0 large_final -1 freeze_init none t`

- Freeze pruned weights that increased in magnitude at initial values: `./print_train_lottery_iterative_command.sh fc test 0 large_final -1 freeze_init_zero_mask none t`

- Initialize weights that decreased in magnitude at 0, and freeze pruned weights at initial value: `./print_train_lottery_iterative_command.sh fc test 0 large_final -1 freeze_init_zero_all none t`

#### Supermask experiments
- Evaluate the initial test accuracy of all alternative mask criteria: `python get_init_loss_train_lottery.py --output_dir ./results/iter_lot_fc_orig/test_seed_0/ --train_h5 ./data/mnist_train.h5 --test_h5 ./data/mnist_test.h5 --arch fc_lot --seed 0 --opt adam --lr 0.0012 --exp none --layer_cutoff 4,6 --prune_base 0.8,0.9 --prune_power 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24`

- Train a Supermask directly: `python train_supermask.py --output_dir ./results/iter_lot_fc_orig/learned_supermasks/run1/ --train_h5 ./data/mnist_train.h5 --test_h5 ./data/mnist_test.h5 --arch fc_mask --opt sgd --lr 100 --num_epochs 2000 --print_every 220 --eval_every 220 --log_every 220 --save_weights --save_every 22000`
