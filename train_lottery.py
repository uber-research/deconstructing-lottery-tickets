
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import time
import h5py
import argparse
import os
import sys
from math import ceil, floor
from ast import literal_eval
from random import shuffle

import network_builders
from tf_plus import Conv2D, MaxPooling2D, Flatten, Dense, relu, Activation
from tf_plus import Layers, SequentialNetwork, l2reg
from tf_plus import learning_phase, batchnorm_learning_phase
from tf_plus import add_classification_losses
from tf_plus import hist_summaries_train, get_collection_intersection, get_collection_intersection_summary, log_scalars, sess_run_dict
from tf_plus import summarize_weights, summarize_opt, tf_assert_all_init, tf_get_uninitialized_variables, add_grad_summaries

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_h5', type=str, required=True)
    parser.add_argument('--test_h5', type=str, required=True)
    parser.add_argument('--val_h5', type=str, required=True)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--save_weights', action='store_true', help='save gradients and weights to file')
    parser.add_argument('--save_loss', action='store_true', help='save loss and accuracy to file')
    parser.add_argument('--input_dim', type=str, default='28,28,1', help='mnist: 28,28,1; cifar: 32,32,3')
    parser.add_argument('--arch', type=str, default='fc_lot', choices=('fc_lot', 'conv2_lot', 'conv4_lot', 'conv6_lot'), help='network architecture')

    #optimization params
    parser.add_argument('--opt', type=str, default='sgd', choices=('sgd', 'rmsprop', 'adam'))
    parser.add_argument('--lr', type=float, default=.01, help='suggested: .01 sgd, .001 rmsprop, .0001 adam')
    parser.add_argument('--decay_lr', action='store_true', help='decay learning rate')
    parser.add_argument('--decay_schedule', type=str, default = '10,20,50,-1', help = 'comma-separated list')
    parser.add_argument('--mom', type=float, default=.9, help='momentum (only has effect for sgd/rmsprop)')
    parser.add_argument('--l2', type=float, default=0)

    parser.add_argument('--num_epochs', type=int, default=1, help='number of epochs')
    parser.add_argument('--train_batch_size', type=int, default=60)
    parser.add_argument('--large_batch_size', type=int, default=5000, help='mnist: 11000, cifar: 5000')
    parser.add_argument('--test_batch_size', type=int, default=0) # do 0 for all
    parser.add_argument('--val_batch_size', type=int, default=0) # do 0 for all
    parser.add_argument('--no_shuffle', action='store_true')

    parser.add_argument('--print_every', type=int, default=100, help='print status update every n iterations')
    parser.add_argument('--output_dir', type=str, default=os.environ.get('GIT_RESULTS_MANAGER_DIR', None), help='output directory')
    parser.add_argument('--output_dir_orig', type=str, help = 'alternative directory to feed in masks')
    parser.add_argument('--eval_every', type=int, default=1, help='eval on entire set')
    parser.add_argument('--log_every', type=int, default=5, help='save tb batch acc/loss every n iterations')
    parser.add_argument('--mode', type = str, default = 'save_all', choices = ('save_all', 'save_res'))
    parser.add_argument('--save_special_iter', type = int, default = -1, help='special iteration to save weights on')
    
    #lottery params
    parser.add_argument('--exp', type=str, default='none', choices = ('none', 'random_reinit', 'random_reshuffle', 'signed_reinit', 'signed_reshuffle', 'signed_constant', 'rand_signed_constant', 'mask_init', 'random_reinit_rescaling', 'signed_reinit_rescaling', 'signed_constant_rescaling', 'rand_signed_constant_rescaling'), help='experiment scenarios')
    
    parser.add_argument('--signed_constant', type=float)
    
    parser.add_argument('--model_type', type=str, default = 'mask', choices = ('mask', 'original', 'freeze_init', 'freeze_reinit', 'freeze_init_zero_mask', 'freeze_init_zero_all', 'freeze_init_zero_reshuffle', 'freeze_init_zero_rev'))
    
    parser.add_argument('--orig_weights', type=str)
    parser.add_argument('--prev_weights', type=str, required=True)
    
    parser.add_argument('--method', type=str, required=True, choices = ('large_final', 'small_final', 'random', 'large_init', 'small_init', 'large_init_large_final', 'small_init_small_final', 'magnitude_increase', 'movement','snip','grad', 'large_final_diff_sign', 'large_final_same_sign', 'magnitude_increase_diff_sign', 'magnitude_increase_same_sign', 'large_final_diff_sign_single', 'large_final_same_sign_single'))
    
    parser.add_argument('--prune_base', type=str, required=True)
    parser.add_argument('--prune_power', type=int, required=True)
    parser.add_argument('--layer_cutoff', type=str, required=True)
    parser.add_argument('--final_weight_interpolation', type=float, default = 1)
    parser.add_argument('--final_weights_ind', type=int, default = -1)

    parser.add_argument('--skip_calc_mask', action='store_true')
    parser.add_argument('--mask_type', type=str, default = 'lottery', choices = ('lottery', 'learned'))
    parser.add_argument('--dynamic_scaling', action='store_true')
    parser.add_argument('--learned_mask_dir', type=str)
    parser.add_argument('--learned_mask_iter', type=int, default=-1)
    return parser

def read_input_data(filename):
    input_file = h5py.File(filename, 'r')
    x = np.array(input_file.get('images'))
    y = np.array(input_file.get('labels'))
    input_file.close()
    return x, y

def sigmoid(x):
    return np.where(x >= 0, 
                    1 / (1 + np.exp(-x)), 
                    np.exp(x) / (1 + np.exp(x)))

################# model setup, after architecture is already created

def init_model(model, input_dim):
    img_size = tuple([None] + [int(dim) for dim in input_dim.split(',')])
    input_images = tf.placeholder(dtype='float32', shape=img_size)
    input_labels = tf.placeholder(dtype='int64', shape=(None,))
    #adding things to trackable
    model.a('input_images', input_images)
    model.a('input_labels', input_labels)
    model.a('logits', model(input_images)) # logits is y_pred

def define_training(model, args):
    # define optimizer
    input_lr = tf.placeholder(tf.float32, shape=[]) # placeholder for dynamic learning rate
    model.a('input_lr', input_lr)
    if args.opt == 'sgd':
        optimizer = tf.train.MomentumOptimizer(input_lr, args.mom)
    elif args.opt == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(input_lr, momentum=args.mom)
    elif args.opt == 'adam':
        optimizer = tf.train.AdamOptimizer(input_lr)
    model.a('optimizer', optimizer)

    # This adds prob, cross_ent, loss_cross_ent, class_prediction, 
    # prediction_correct, accuracy, loss, (loss_reg) in tf_nets/losses.py
    add_classification_losses(model, model.input_labels)
    
    grads_and_vars = optimizer.compute_gradients(model.loss, model.trainable_weights, gate_gradients=tf.train.Optimizer.GATE_GRAPH)
    model.a('grads_to_compute', [grad for grad, _ in grads_and_vars])
    model.a('train_step', optimizer.apply_gradients(grads_and_vars))

    print('All model weights:')
    summarize_weights(model.trainable_weights) #print summaries for weights (from tfutil)
    # print('grad summaries:')
    # add_grad_summaries(grads_and_vars)
    # print('opt summary:')
    # summarize_opt(optimizer)

################# util for training/eval portion

# flatten and concatentate list of tensors into one np vector
def flatten_all(tensors):
    return np.concatenate([tensor.eval().flatten() for tensor in tensors])


def split_and_shape(one_time_slice, shapes):
    variables = []
    offset = 0
    for shape in shapes:
        num_params = np.prod(shape)
        variables.append(one_time_slice[offset : offset + num_params].reshape(shape))
        offset += num_params
    return variables

# eval on whole train/test set occasionally, for tuning purposes
def eval_on_entire_dataset(sess, model, input_x, input_y, dim_sum, batch_size, tb_prefix, tb_writer, iterations):
    grad_sums = np.zeros(dim_sum)
    num_batches = int(input_y.shape[0] / batch_size)
    total_acc = 0
    total_loss = 0
    total_loss_no_reg = 0 # loss without counting l2 penalty

    for i in range(num_batches):
        # slice indices (should be large)
        s_start = batch_size * i
        s_end = s_start + batch_size

        fetch_dict = {
            'accuracy': model.accuracy,
            'loss': model.loss,
            'loss_no_reg': model.loss_cross_ent}

        #sess_run_dict is from tfutil and it returns a dictionary
        result_dict = sess_run_dict(sess, fetch_dict, feed_dict={
            model.input_images: input_x[s_start:s_end], 
            model.input_labels: input_y[s_start:s_end],
            learning_phase(): 0,
            batchnorm_learning_phase(): 1}) # do not use nor update moving averages (****??****)

        total_acc += result_dict['accuracy']
        total_loss += result_dict['loss']
        total_loss_no_reg += result_dict['loss_no_reg']

    acc = total_acc / num_batches
    loss = total_loss / num_batches
    loss_no_reg = total_loss_no_reg / num_batches

    # tensorboard
    if tb_writer:
        summary = tf.Summary()
        summary.value.add(tag='%s_acc' % tb_prefix, simple_value=acc)
        summary.value.add(tag='%s_loss' % tb_prefix, simple_value=loss)
        summary.value.add(tag='%s_loss_no_reg' % tb_prefix, simple_value=loss_no_reg)
        tb_writer.add_summary(summary, iterations)

    return acc, loss_no_reg

#################


def train_and_eval(sess, model, train_x, train_y, val_x, val_y, test_x, test_y, tb_writer, dsets, args):
    # constants
    num_batches = int(train_y.shape[0] / args.train_batch_size)
    dim_sum = sum([tf.size(var).eval() for var in model.trainable_weights]) #dimention of weight matrices

    # adaptive learning schedule
    curr_lr = args.lr
    decay_schedule = [int(x) for x in args.decay_schedule.split(',')]
    print(decay_schedule)
    decay_count = 0

    # initializations
    tb_summaries = tf.summary.merge(tf.get_collection('train_step'))

    shuffled_indices = np.arange(train_y.shape[0]) # for no shuffling
    iterations = 0
    chunks_written = 0 
    timerstart = time.time()
    iter_index = 0
    
    if args.save_weights:
        dsets['all_weights'][chunks_written] = flatten_all(model.trainable_weights)
                
    chunks_written += 1
        
    for epoch in range(args.num_epochs):
        if not args.no_shuffle:
            shuffled_indices = np.random.permutation(train_y.shape[0]) # for shuffled mini-batches

        if args.decay_lr and epoch == decay_schedule[decay_count]:
            curr_lr *= 0.1
            decay_count += 1
            print('dropping learning rate to ' + str(curr_lr))

        for i in range(num_batches):
            
            # less frequent, larger evals
            if iterations % args.eval_every == 0:
                # eval on entire train set
                cur_train_acc, cur_train_loss = eval_on_entire_dataset(sess, model, train_x, train_y,
                    dim_sum, args.large_batch_size, 'eval_train', tb_writer, iterations)

                # eval on entire test/val set
                cur_test_acc, cur_test_loss = eval_on_entire_dataset(sess, model, test_x, test_y,
                    dim_sum, args.test_batch_size, 'eval_test', tb_writer, iterations)
                
                cur_val_acc, cur_val_loss = eval_on_entire_dataset(sess, model, val_x, val_y,
                    dim_sum, args.val_batch_size, 'eval_val', tb_writer, iterations)
                
                if args.save_loss:
                    dsets['train_accuracy'][iter_index] = cur_train_acc
                    dsets['train_loss'][iter_index] = cur_train_loss
                    dsets['val_accuracy'][iter_index] = cur_val_acc
                    dsets['val_loss'][iter_index] = cur_val_loss
                    dsets['test_accuracy'][iter_index] = cur_test_acc
                    dsets['test_loss'][iter_index] = cur_test_loss
                    iter_index += 1
                
            # print status update
            if iterations % args.print_every == 0:
                print(('{}: train acc = {:.4f}, val acc = {:.4f}, test acc = {:.4f}, '
                    + 'train loss = {:.4f}, val loss = {:.4f}, test loss = {:.4f} ({:.2f} s)').format(iterations,
                    cur_train_acc, cur_val_acc, cur_test_acc, cur_train_loss, cur_val_loss, cur_test_loss, time.time() - timerstart))

            # current slice for input data
            batch_indices = shuffled_indices[args.train_batch_size * i : args.train_batch_size * (i + 1)]

            # training
            fetch_dict = {'train_step': model.train_step,
                'accuracy': model.accuracy,
                'loss': model.loss}
            fetch_dict.update(model.update_dict())
            if iterations % args.log_every == 0:
                fetch_dict.update({'tb': tb_summaries})
            result_train = sess_run_dict(sess, fetch_dict, feed_dict={
                model.input_images: train_x[batch_indices],
                model.input_labels: train_y[batch_indices],
                model.input_lr: curr_lr,
                learning_phase(): 1,
                batchnorm_learning_phase(): 1})

            # log to tensorboard
            if tb_writer and iterations % args.log_every == 0:
                tb_writer.add_summary(result_train['tb'], iterations)

            iterations += 1

            if iterations == 1:
                dsets['all_weights'][chunks_written] = flatten_all(model.trainable_weights)
                chunks_written += 1
                
            if iterations == args.save_special_iter:
                dsets['all_weights'][chunks_written] = flatten_all(model.trainable_weights)
                chunks_written += 1
                
            # store current weights and gradients
            if args.mode == 'save_all' and  args.save_weights and iterations % args.eval_every == 0 and iterations != args.save_special_iter:
                dsets['all_weights'][chunks_written] = flatten_all(model.trainable_weights)
                chunks_written += 1
                    

    # save final weight values
    if args.save_weights and (iterations % args.eval_every != 0 or args.mode == 'save_res'):
        dsets['all_weights'][chunks_written] = flatten_all(model.trainable_weights)

    # save final evals
    # on entire train set
    cur_train_acc, cur_train_loss = eval_on_entire_dataset(sess, model, train_x, train_y,
        dim_sum, args.large_batch_size, 'eval_train', tb_writer, iterations)
    
    # on entire test/val set
    cur_test_acc, cur_test_loss = eval_on_entire_dataset(sess, model, test_x, test_y,
        dim_sum, args.test_batch_size, 'eval_test', tb_writer, iterations)
    
    cur_val_acc, cur_val_loss = eval_on_entire_dataset(sess, model, val_x, val_y,
        dim_sum, args.val_batch_size, 'eval_val', tb_writer, iterations)

    if args.save_loss and iterations % args.eval_every != 0:
        dsets['train_accuracy'][iter_index] = cur_train_acc
        dsets['train_loss'][iter_index] = cur_train_loss
        dsets['test_accuracy'][iter_index] = cur_test_acc
        dsets['test_loss'][iter_index] = cur_test_loss
        dsets['val_accuracy'][iter_index] = cur_val_acc
        dsets['val_loss'][iter_index] = cur_val_loss

    # print last status update
    print(('{}: train acc = {:.4f}, val acc = {:.4f}, test acc = {:.4f}, '
        + 'train loss = {:.4f}, val loss = {:.4f}, test loss = {:.4f} ({:.2f} s)').format(iterations,
        cur_train_acc, cur_val_acc, cur_test_acc, cur_train_loss, cur_val_loss, cur_test_loss, time.time() - timerstart))

    

def main():
    parser = make_parser()
    args = parser.parse_args()
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)
    
    #calculate mask data
    if not args.skip_calc_mask:
        os.system('python get_weight_init.py --method ' + args.method + ' --weights_h5 ' + args.prev_weights + '/weights --output_h5 ' + args.output_dir + '/init_weights --prune_base ' + args.prune_base + ' --prune_power ' + str(args.prune_power) + ' --layer_cutoff ' + args.layer_cutoff + ' --seed ' + str(args.seed) + ' --final_weight_interpolation ' + str(args.final_weight_interpolation) + ' --final_weights_ind ' + str(args.final_weights_ind))

    #load mask data
    if args.mask_type == 'lottery':
        if not args.skip_calc_mask:
            hf = h5py.File(args.output_dir + '/init_weights', 'r')
        else:
            hf = h5py.File(args.output_dir_orig + '/init_weights', 'r')
        shapes = [literal_eval(s) for s in hf.attrs['var_shapes'].decode('utf-8').split(';')]
        initial_weights = np.array(hf.get('initial_weights'))
        mask_values = np.array(hf.get('mask_values'))

        final_weights = np.array(hf.get('final_weights'))
        final_weights = split_and_shape(final_weights, shapes)

        hf.close()

        initial_weights = split_and_shape(initial_weights, shapes)
        mask_values = split_and_shape(mask_values, shapes)

    elif args.mask_type == 'learned' and args.skip_calc_mask:
        hf = h5py.File(args.learned_mask_dir + '/weights', 'r')
        
        shapes = [literal_eval(s) for s in hf.attrs['var_shapes'].decode('utf-8').split(';')]
        layer_names = [s for s in hf.attrs['var_names'].decode('utf-8').split(';')]
        
        init_all_weights = np.array(hf.get('all_weights')[0])
        final_all_weights = np.array(hf.get('all_weights')[args.learned_mask_iter])
        
        hf.close()
        
        init_all_weights = split_and_shape(init_all_weights, shapes)
        final_all_weights = split_and_shape(final_all_weights, shapes)
        print(layer_names)
        initial_weights = [init_all_weights[i] for i, x in enumerate(layer_names) if 'mask' not in x]
        mask_values = [np.random.binomial(n = 1, p = sigmoid(final_all_weights[i])) if 'mask' in x else np.ones(final_all_weights[i].shape, dtype=np.int) if 'bias' in x else None for i, x in enumerate(layer_names) if 'bias' in x or 'mask' in x]
        mask_values = [mask_values[i] for i in np.tile([1,-1], len(mask_values)//2) + np.arange(len(mask_values))]
        
        print('initial weights num is %d' % len(initial_weights))
        print('mask values num is %d' % len(mask_values))
        if args.dynamic_scaling:
            dynamic_scaling_multiplier = [mask_values[i].size / np.sum(mask_values[i]) for i in range(len(mask_values))]
            initial_weights = [initial_weights[i] * dynamic_scaling_multiplier[i] for i in range(len(initial_weights))]
            
    signed_constants = []
    
    
    if args.model_type == 'freeze_init':
        freeze_init_value = initial_weights[:]
        
        
    if args.model_type == 'freeze_init_zero_mask' or args.exp == 'freeze_init_zero_mask':
        freeze_init_value = initial_weights[:]
        for i in range(len(initial_weights)):
            temp = freeze_init_value[i].flatten()
            init_temp = np.abs(initial_weights[i].flatten())
            final_temp = np.abs(final_weights[i].flatten())
            temp[np.squeeze(np.where(init_temp > final_temp))] = 0 
            freeze_init_value[i][~mask_values[i].astype('bool')] = temp.reshape(initial_weights[i].shape)[~mask_values[i].astype('bool')]
            initial_weights[i] = freeze_init_value[i]
    

    if args.model_type == 'freeze_init_zero_all' or args.exp == 'freeze_init_zero_all':
        freeze_init_value = initial_weights[:]
        for i in range(len(initial_weights)):
            temp = freeze_init_value[i].flatten()
            init_temp = np.abs(initial_weights[i].flatten())
            final_temp = np.abs(final_weights[i].flatten())
            temp[np.squeeze(np.where(init_temp > final_temp))] = 0 
            freeze_init_value[i] = temp.reshape(initial_weights[i].shape)
            initial_weights[i] = freeze_init_value[i]
            

    if args.model_type == 'freeze_init_zero_reshuffle':
        freeze_init_value = initial_weights[:]
        for i in range(len(initial_weights)):
            temp = freeze_init_value[i][~mask_values[i].astype('bool')]
            init_temp = np.abs(initial_weights[i][~mask_values[i].astype('bool')])
            final_temp = np.abs(final_weights[i][~mask_values[i].astype('bool')])
            zero_size = np.squeeze(np.where(init_temp > final_temp)).size
            if zero_size > 0:
                temp[np.random.choice(range(len(init_temp)), zero_size, replace = False)] = 0
            freeze_init_value[i][~mask_values[i].astype('bool')] = temp
            initial_weights[i] = freeze_init_value[i]
            
    if args.model_type == 'freeze_init_zero_rev':
        freeze_init_value = initial_weights[:]
        for i in range(len(initial_weights)):
            temp = freeze_init_value[i].flatten()
            init_temp = np.abs(initial_weights[i].flatten())
            final_temp = np.abs(final_weights[i].flatten())
            temp[np.squeeze(np.where(init_temp < final_temp))] = 0 
            freeze_init_value[i][~mask_values[i].astype('bool')] = temp.reshape(initial_weights[i].shape)[~mask_values[i].astype('bool')]
            initial_weights[i] = freeze_init_value[i]
    
    #load data
    train_x, train_y = read_input_data(args.train_h5)
    val_x, val_y = read_input_data(args.val_h5)
    test_x, test_y = read_input_data(args.test_h5)

    images_scale = np.max(train_x)
    if images_scale > 1:
        print('Normalizing images by a factor of {}'.format(images_scale))
        train_x = train_x / images_scale
        val_x = val_x / images_scale
        test_x = test_x / images_scale

    if args.test_batch_size == 0:
        args.test_batch_size = test_y.shape[0]
        
    if args.val_batch_size == 0:
        args.val_batch_size = val_y.shape[0]

    print('Data shapes:', train_x.shape, train_y.shape, test_x.shape, test_y.shape)
    if train_y.shape[0] % args.train_batch_size != 0:
        print("WARNING batch size doesn't divide train set evenly")
    if train_y.shape[0] % args.large_batch_size != 0:
        print("WARNING large batch size doesn't divide train set evenly")
    if test_y.shape[0] % args.test_batch_size != 0:
        print("WARNING batch size doesn't divide test set evenly")
    if val_y.shape[0] % args.val_batch_size != 0:
        print("WARNING batch size doesn't divide validation set evenly")
        
    if 'mnist' in args.train_h5:
        input_dim = '28,28,1'
    elif 'cifar10' in args.train_h5:
        input_dim = '32,32,3'

    #build model 
    if args.model_type == 'mask':
        if args.arch == 'fc_lot':
            model = network_builders.build_masked_fc_lottery(args, mask_values)
        elif args.arch == 'conv2_lot':
            model = network_builders.build_masked_conv2_lottery(args, mask_values)
        elif args.arch == 'conv4_lot':
            model = network_builders.build_masked_conv4_lottery(args, mask_values)
        elif args.arch == 'conv6_lot':
            model = network_builders.build_masked_conv6_lottery(args, mask_values)
            
    elif args.model_type == 'original':
        if args.arch == 'fc_lot':
            model = network_builders.build_fc_lottery(args)
        elif args.arch == 'conv2_lot':
            model = network_builders.build_conv2_lottery(args)
        elif args.arch == 'conv4_lot':
            model = network_builders.build_conv4_lottery(args)
        elif args.arch == 'conv6_lot':
            model = network_builders.build_conv6_lottery(args)
            
    else:
        if args.arch == 'fc_lot':
            model = network_builders.build_frozen_fc_lottery(args, freeze_init_value, mask_values)
        elif args.arch == 'conv2_lot':
            model = network_builders.build_frozen_conv2_lottery(args, freeze_init_value, mask_values)
        elif args.arch == 'conv4_lot':
            model = network_builders.build_frozen_conv4_lottery(args, freeze_init_value, mask_values)
        elif args.arch == 'conv6_lot':
            model = network_builders.build_frozen_conv6_lottery(args, freeze_init_value, mask_values)
            
    init_model(model, input_dim)
    define_training(model, args)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    
    if args.exp == 'none':
        for i, w in enumerate(model.trainable_weights):
            w.load(initial_weights[i], session=sess)
            
    elif args.exp == 'mask_init':
        for i in range(len(initial_weights)):
            initial_weights[i][~mask_values[i].astype('bool')] = 0
            
        for i, w in enumerate(model.trainable_weights):
            w.load(initial_weights[i], session=sess)
            
    elif args.exp == 'random_reshuffle':
        for i in range(len(initial_weights)):
            temp = initial_weights[i][mask_values[i].astype('bool')]
            shuffle(temp)
            initial_weights[i][mask_values[i].astype('bool')] = temp
            
        for i, w in enumerate(model.trainable_weights):
            w.load(initial_weights[i], session=sess)
            
    elif args.exp == 'signed_reshuffle':
        for i in range(len(initial_weights)):
            temp = initial_weights[i][mask_values[i].astype('bool')]
            shuffle(temp)
            temp = np.abs(temp)
            init_sign = np.sign(initial_weights[i][mask_values[i].astype('bool')])
            temp = temp * init_sign
            initial_weights[i][mask_values[i].astype('bool')] = temp
            
        for i, w in enumerate(model.trainable_weights):
            w.load(initial_weights[i], session=sess)

    elif args.exp == 'random_reinit':
        for i in range(len(initial_weights)):
            temp = initial_weights[i].flatten()
            shuffle(temp)
            initial_weights[i] = temp.reshape(initial_weights[i].shape)
            
        for i, w in enumerate(model.trainable_weights):
            w.load(initial_weights[i], session=sess)
            

    elif args.exp == 'signed_reinit':
        for i in range(len(initial_weights)):
            temp = initial_weights[i].flatten()
            shuffle(temp)
            temp = np.abs(temp).reshape(initial_weights[i].shape)
            init_sign = np.sign(initial_weights[i])
            temp = temp * init_sign
            initial_weights[i] = temp
            
        for i, w in enumerate(model.trainable_weights):
            w.load(initial_weights[i], session=sess)
            
    elif args.exp == 'random_reinit_rescaling':
        hfo = h5py.File(args.orig_weights + '/weights', 'r')
        orig_initial_weights = np.array(hfo.get('all_weights')[0])
        hfo.close()
        initial_weights = split_and_shape(orig_initial_weights, shapes)
        
        for i in range(len(initial_weights)):
            temp = initial_weights[i].flatten()
            shuffle(temp)
            initial_weights[i] = temp.reshape(initial_weights[i].shape)
            
            ones_in_mask = np.sum(mask_values[i])
            multiplier = mask_values[i].size / ones_in_mask
            print('multiplier is %d' %multiplier)
            initial_weights[i] = initial_weights[i] * multiplier
            
        for i, w in enumerate(model.trainable_weights):
            w.load(initial_weights[i], session=sess)
            
    elif args.exp == 'signed_reinit_rescaling':
        hfo = h5py.File(args.orig_weights + '/weights', 'r')
        orig_initial_weights = np.array(hfo.get('all_weights')[0])
        hfo.close()
        initial_weights = split_and_shape(orig_initial_weights, shapes)
        
        for i in range(len(initial_weights)):
            temp = initial_weights[i].flatten()
            shuffle(temp)
            temp = np.abs(temp).reshape(initial_weights[i].shape)
            init_sign = np.sign(initial_weights[i])
            temp = temp * init_sign
            initial_weights[i] = temp
            
            ones_in_mask = np.sum(mask_values[i])
            multiplier = mask_values[i].size / ones_in_mask
            print('multiplier is %d' %multiplier)
            initial_weights[i] = initial_weights[i] * multiplier
            
        for i, w in enumerate(model.trainable_weights):
            w.load(initial_weights[i], session=sess)
            
            
    elif args.exp == 'rand_signed_constant':
        for i in range(len(initial_weights)):
            if args.signed_constant is None:
                signed_constant_val = np.std(initial_weights[i])
            else:
                signed_constant_val = args.signed_constant
            
            init_sign = np.sign(initial_weights[i])
            temp = signed_constant_val * init_sign 
            temp = temp.flatten()
            shuffle(temp)
            initial_weights[i] = temp.reshape(initial_weights[i].shape)
            signed_constants.append(signed_constant_val)
            
        for i, w in enumerate(model.trainable_weights):
            w.load(initial_weights[i], session=sess)
            
            
    elif args.exp == 'signed_constant':
        for i in range(len(initial_weights)):
            if args.signed_constant is None:
                signed_constant_val = np.std(initial_weights[i])
            else:
                signed_constant_val = args.signed_constant
            init_sign = np.sign(initial_weights[i])
            temp = signed_constant_val * init_sign 
            initial_weights[i] = temp
            signed_constants.append(signed_constant_val)
            
        for i, w in enumerate(model.trainable_weights):
            w.load(initial_weights[i], session=sess)
            
    elif args.exp == 'rand_signed_constant_rescaling':
        hfo = h5py.File(args.orig_weights + '/weights', 'r')
        orig_initial_weights = np.array(hfo.get('all_weights')[0])
        hfo.close()
        initial_weights = split_and_shape(orig_initial_weights, shapes)
        
        for i in range(len(initial_weights)):
            if args.signed_constant is None:
                signed_constant_val = np.std(initial_weights[i])
            else:
                signed_constant_val = args.signed_constant
            
            init_sign = np.sign(initial_weights[i])
            temp = signed_constant_val * init_sign 
            temp = temp.flatten()
            shuffle(temp)
            initial_weights[i] = temp.reshape(initial_weights[i].shape)
            signed_constants.append(signed_constant_val)
            
            ones_in_mask = np.sum(mask_values[i])
            multiplier = mask_values[i].size / ones_in_mask
            print('multiplier is %d' %multiplier)
            initial_weights[i] = initial_weights[i] * multiplier
            
            
        for i, w in enumerate(model.trainable_weights):
            w.load(initial_weights[i], session=sess)
            
            
    elif args.exp == 'signed_constant_rescaling':
        hfo = h5py.File(args.orig_weights + '/weights', 'r')
        orig_initial_weights = np.array(hfo.get('all_weights')[0])
        hfo.close()
        initial_weights = split_and_shape(orig_initial_weights, shapes)
        
        for i in range(len(initial_weights)):
            if args.signed_constant is None:
                signed_constant_val = np.std(initial_weights[i])
            else:
                signed_constant_val = args.signed_constant
            init_sign = np.sign(initial_weights[i])
            temp = signed_constant_val * init_sign 
            initial_weights[i] = temp
            signed_constants.append(signed_constant_val)
            
            ones_in_mask = np.sum(mask_values[i])
            multiplier = mask_values[i].size / ones_in_mask
            print('multiplier is %d' %multiplier)
            initial_weights[i] = initial_weights[i] * multiplier
            
        for i, w in enumerate(model.trainable_weights):
            w.load(initial_weights[i], session=sess)
            


    for collection in ['train_step']: # 'eval_train' and 'eval_test' added manually later
        tf.summary.scalar(collection + '_acc', model.accuracy, collections=[collection])
        tf.summary.scalar(collection + '_loss', model.loss, collections=[collection])

    tb_writer, hf = None, None
    dsets = {}
    if args.output_dir:
        tb_writer = tf.summary.FileWriter(args.output_dir, sess.graph)
    
        # set up output for gradients/weights
        if args.save_weights:
            dim_sum = sum([tf.size(var).eval() for var in model.trainable_weights])
            total_iters = args.num_epochs * int(train_y.shape[0] / args.train_batch_size)
            if args.mode == 'save_all':
                total_chunks = int(ceil(total_iters / args.eval_every))
            elif args.mode == 'save_res':
                if args.save_special_iter == -1:
                    total_chunks = 1
                else:
                    total_chunks = 2
            hf = h5py.File(args.output_dir + '/weights', 'w-')

            # write metadata
            var_shapes = np.string_(';'.join([str(var.get_shape()) for var in model.trainable_weights]))
            hf.attrs['var_shapes'] = var_shapes
            var_names = np.string_(';'.join([str(var.name) for var in model.trainable_weights]))
            hf.attrs['var_names'] = var_names
            hf.attrs['prune_base'] = args.prune_base
            hf.attrs['prune_power'] = args.prune_power
#             hf.attrs['signed_constant_multiplier'] = args.signed_constant_multiplier
            
            dsets['all_weights'] = hf.create_dataset('all_weights', (total_chunks + 2, dim_sum), dtype='f8', compression='gzip')
            
            #save masks
            mask_values = np.concatenate([x.flatten() for x in mask_values])
            dsets['mask_values'] = hf.create_dataset('mask_values', data=mask_values)
            dsets['signed_constants'] = hf.create_dataset('signed_constants', data=np.array(signed_constants))
            
        if args.save_loss:
            dsets['train_accuracy'] = hf.create_dataset('train_accuracy', (int(ceil(total_iters / args.eval_every)) + 1, 1), dtype='f8', compression='gzip')
            dsets['train_loss'] = hf.create_dataset('train_loss', (int(ceil(total_iters / args.eval_every)) + 1, 1), dtype='f8', compression='gzip')
            dsets['val_accuracy'] = hf.create_dataset('val_accuracy', (int(ceil(total_iters / args.eval_every)) + 1, 1), dtype='f8', compression='gzip')
            dsets['val_loss'] = hf.create_dataset('val_loss', (int(ceil(total_iters / args.eval_every)) + 1, 1), dtype='f8', compression='gzip')
            dsets['test_accuracy'] = hf.create_dataset('test_accuracy', (int(ceil(total_iters / args.eval_every)) + 1, 1), dtype='f8', compression='gzip')
            dsets['test_loss'] = hf.create_dataset('test_loss', (int(ceil(total_iters / args.eval_every)) + 1, 1), dtype='f8', compression='gzip')
            
    print(signed_constants)
    
    ########## Run main thing ##########
    train_and_eval(sess, model, train_x, train_y, val_x, val_y, test_x, test_y, tb_writer, dsets, args)

    if tb_writer:
        tb_writer.close()
    if hf:
        hf.close()

if __name__ == '__main__':
    main()

