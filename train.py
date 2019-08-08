# Copyright (c) 2019 Uber Technologies, Inc.

# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project. 

# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import time
import h5py
import argparse
import os
import sys
from math import ceil

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
    parser.add_argument('--eval_every', type=int, default=1, help='eval on entire set')
    parser.add_argument('--log_every', type=int, default=5, help='save tb batch acc/loss every n iterations')

    parser.add_argument('--mode', type = str, default = 'save_all', choices = ('save_all', 'save_res'))
    return parser

def read_input_data(filename):
    input_file = h5py.File(filename, 'r')
    x = np.array(input_file.get('images'))
    y = np.array(input_file.get('labels'))
    input_file.close()
    return x, y

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


def train_and_eval(sess, model, snip_batch_size, train_x, train_y, val_x, val_y, test_x, test_y, tb_writer, dsets, args):
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
        
    dsets['one_iter_grads'][0] = calc_one_iter_grads(sess, model, train_x, train_y, snip_batch_size, dsets)

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
                
            # store current weights and gradients
            if args.mode == 'save_all' and  args.save_weights and iterations % args.eval_every == 0:
                dsets['all_weights'][chunks_written] = flatten_all(model.trainable_weights)
                chunks_written += 1
                    

    # save final weight values
    if args.save_weights and iterations % args.eval_every != 0:
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

# loads weights, calculates train and test gradients, writes to file at given iteration 
def calc_one_iter_grads(sess, model, train_x, train_y, snip_batch_size, dsets):
    train_size = train_x.shape[0]
    batch_ind = np.random.choice(range(train_size), size=snip_batch_size, replace=False)
    fetch_dict = {}
    fetch_dict['gradients'] = model.grads_to_compute
    
    result_dict = sess_run_dict(sess, fetch_dict, feed_dict={
            model.input_images: train_x[batch_ind], 
            model.input_labels: train_y[batch_ind],
            learning_phase(): 0,
            batchnorm_learning_phase(): 1})
    
    grads = result_dict['gradients']
    flattened = np.concatenate([grad.flatten() for grad in grads])

    return flattened


def main():
    parser = make_parser()
    args = parser.parse_args()
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)
    
#     load data
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
        snip_batch_size = 100
    elif 'cifar10' in args.train_h5:
        input_dim = '32,32,3'
        snip_batch_size = 128

    # build model 
    if args.arch == 'fc_lot':
        model = network_builders.build_fc_lottery(args)
    elif args.arch == 'conv2_lot':
        model = network_builders.build_conv2_lottery(args)
    elif args.arch == 'conv4_lot':
        model = network_builders.build_conv4_lottery(args)
    elif args.arch == 'conv6_lot':
        model = network_builders.build_conv6_lottery(args)
        
    init_model(model, input_dim)
    define_training(model, args)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

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
                total_chunks = 1
            hf = h5py.File(args.output_dir + '/weights', 'w-')

            # write metadata
            var_shapes = np.string_(';'.join([str(var.get_shape()) for var in model.trainable_weights]))
            hf.attrs['var_shapes'] = var_shapes
            var_names = np.string_(';'.join([str(var.name) for var in model.trainable_weights]))
            hf.attrs['var_names'] = var_names
            
            dsets['all_weights'] = hf.create_dataset('all_weights', (total_chunks + 2, dim_sum), dtype='f8', compression='gzip')
            dsets['one_iter_grads'] = hf.create_dataset('one_iter_grads', (1, dim_sum), dtype='f8', compression='gzip')

        if args.save_loss:
            dsets['train_accuracy'] = hf.create_dataset('train_accuracy', (int(ceil(total_iters / args.eval_every)) + 1, 1), dtype='f8', compression='gzip')
            dsets['train_loss'] = hf.create_dataset('train_loss', (int(ceil(total_iters / args.eval_every)) + 1, 1), dtype='f8', compression='gzip')
            dsets['val_accuracy'] = hf.create_dataset('val_accuracy', (int(ceil(total_iters / args.eval_every)) + 1, 1), dtype='f8', compression='gzip')
            dsets['val_loss'] = hf.create_dataset('val_loss', (int(ceil(total_iters / args.eval_every)) + 1, 1), dtype='f8', compression='gzip')
            dsets['test_accuracy'] = hf.create_dataset('test_accuracy', (int(ceil(total_iters / args.eval_every)) + 1, 1), dtype='f8', compression='gzip')
            dsets['test_loss'] = hf.create_dataset('test_loss', (int(ceil(total_iters / args.eval_every)) + 1, 1), dtype='f8', compression='gzip')

    ########## Run main thing ##########
    train_and_eval(sess, model, snip_batch_size, train_x, train_y, val_x, val_y, test_x, test_y, tb_writer, dsets, args)

    if tb_writer:
        tb_writer.close()
    if hf:
        hf.close()

if __name__ == '__main__':
    main()

