# Copyright (c) 2019 Uber Technologies, Inc.

# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project. 

# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
from __future__ import division

from ast import literal_eval
import tensorflow as tf
import numpy as np
import time
import h5py
import argparse
import os
import sys

import masked_networks
from tf_plus import learning_phase, batchnorm_learning_phase
from tf_plus import sess_run_dict, add_classification_losses
from tf_plus import summarize_weights

def make_parser():
    parser = argparse.ArgumentParser()
    # inputs
    parser.add_argument('--train_h5', type=str, required=True)
    parser.add_argument('--test_h5', type=str, required=True)
    parser.add_argument('--input_dim', type=str, default='28,28,1', help='mnist: 28,28,1; cifar: 32,32,3')
    parser.add_argument('--init_weights_h5', type=str, required=False)
    parser.add_argument('--load', type=str, help='Load checkpoint weight file')
    parser.add_argument('--resume', type=str, help='Load checkpoint weight file and resume training from there')

    # model architecture
    parser.add_argument('--arch', type=str, default='fc', choices=('fc_mask', 'conv2_mask', 'conv4_mask', 'conv6_mask'), help='network architecture')
    
    # training params
    parser.add_argument('--opt', type=str, default='sgd', choices=('sgd', 'rmsprop', 'adam'))
    parser.add_argument('--lr', type=float, default=.01, help='suggested: .01 sgd, .001 rmsprop, .0001 adam')
    parser.add_argument('--decay_schedule', type=str, default='-1', help='comma separated decay learning rate. allcnn: 200,250,300')
    parser.add_argument('--mom', type=float, default=.9, help='momentum (only has effect for sgd/rmsprop)')
    parser.add_argument('--l2', type=float, default=0)
    parser.add_argument('--num_epochs', type=int, default=1, help='number of epochs')
    parser.add_argument('--train_batch_size', type=int, default=250)
    parser.add_argument('--large_batch_size', type=int, default=11000, help='use mnist: 11000, cifar: 5000')
    parser.add_argument('--test_batch_size', type=int, default=0) # do 0 for all
    parser.add_argument('--no_shuffle', action='store_true')
    parser.add_argument('--shuffle_seed', type=int, default=-1, help='seed if you want to shuffle batches')
    parser.add_argument('--tf_seed', type=int, default=-1, help='tensorflow random seed')

    # eval and outputs
    parser.add_argument('--print_every', type=int, default=100, help='print status update every n iterations')
    parser.add_argument('--output_dir', type=str, default=os.environ.get('GIT_RESULTS_MANAGER_DIR', None), help='output directory')
    parser.add_argument('--eval_every', type=int, default=20, help='eval on entire set')
    parser.add_argument('--log_every', type=int, default=5, help='save tb batch acc/loss every n iterations')
    parser.add_argument('--save_weights', action='store_true', help='save gradients and weights to file')
    parser.add_argument('--save_every', type=int, default=1, help='save gradients every n iterations (averaged)') # kinda deprecated

    # supermask configs
    parser.add_argument('--sigmoid_bias', type=float, default=0, help='rounding bias (masks initialized with this)')
    parser.add_argument('--round_mask', action='store_true', help='round masks instead of bernoulli sample')
    parser.add_argument('--signed_constant', action='store_true', help='make network weights signed constant')
    parser.add_argument('--signed_constant_multiplier', type=float, default=1.0, help='Value of multiplier to the default as signed constant (std of each layer init)')
    parser.add_argument('--dynamic_scaling', action='store_true', help='dynamically determine singed constant multiplier based on percentage of masked weights')
    return parser

def read_input_data(filename):
    input_file = h5py.File(filename, 'r')
    x = np.array(input_file['images'])
    y = np.array(input_file['labels'])
    input_file.close()
    return x, y

################# model setup, after architecture is already created

def init_model(model, args):
    img_size = tuple([None] + [int(dim) for dim in args.input_dim.split(',')])
    input_images = tf.placeholder(dtype='float32', shape=img_size)
    input_labels = tf.placeholder(dtype='int64', shape=(None,))
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
    model.a('train_step', optimizer.minimize(model.loss, var_list=model.trainable_weights))
    
    print('All model weights:')
    summarize_weights(model.trainable_weights)
    
################# methods used for freezing layers

# returns list of variables as np arrays in their original shape
def split_and_shape(one_time_slice, shapes):
    variables = []
    offset = 0
    for shape in shapes:
        num_params = np.prod(shape)
        variables.append(one_time_slice[offset : offset + num_params].reshape(shape))
        offset += num_params
    return variables

def load_initial_weights(sess, model, args):
    if not args.init_weights_h5.endswith('/weights'):
        h5file = os.path.join(args.init_weights_h5, 'weights')
    else:
        h5file = args.init_weights_h5
    hf_weights = h5py.File(h5file, 'r')
    init_weights_flat = hf_weights.get('all_weights')[0]
    shapes = [literal_eval(s) for s in hf_weights.attrs['var_shapes'].decode('utf-8').split(';')]
    hf_weights.close()

    weight_values = split_and_shape(init_weights_flat, shapes)
    for i, w in enumerate(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)):
       #if 'mask' not in w.name: # HACK for biased masks
        print('loading weights for layer {}: {}'.format(i, w.name))
        w.load(weight_values[i], session=sess)
    return 



################# util for training/eval portion

# flatten and concatentate list of tensors into one np vector
def flatten_all(tensors):
    return np.concatenate([tensor.eval().flatten() for tensor in tensors])

# eval on whole train/test set occasionally, for tuning purposes
def eval_on_entire_dataset(sess, model, input_x, input_y, batch_size, tb_prefix_and_iter, tb_writer):
    #grad_sums = np.zeros(dim_sum)
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

        result_dict = sess_run_dict(sess, fetch_dict, feed_dict={
            model.input_images: input_x[s_start:s_end], 
            model.input_labels: input_y[s_start:s_end],
            learning_phase(): 0,
            batchnorm_learning_phase(): 1}) # do not use nor update moving averages

        total_acc += result_dict['accuracy']
        total_loss += result_dict['loss']
        total_loss_no_reg += result_dict['loss_no_reg']

    acc = total_acc / num_batches
    loss = total_loss / num_batches
    loss_no_reg = total_loss_no_reg / num_batches

    # tensorboard
    if tb_writer:
        tb_prefix, iterations = tb_prefix_and_iter
        summary = tf.Summary()
        summary.value.add(tag='%s_acc' % tb_prefix, simple_value=acc)
        summary.value.add(tag='%s_loss' % tb_prefix, simple_value=loss)
        summary.value.add(tag='%s_loss_no_reg' % tb_prefix, simple_value=loss_no_reg)
        tb_writer.add_summary(summary, iterations)

    return acc, loss_no_reg

#################


def eval(sess, model, train_x, train_y, test_x, test_y, args, tb_writer, iterations):
    timerstart = time.time()
    # eval on entire train set
    tb_prefix_and_iter = ('eval_train', iterations) if tb_writer else (None, None)
    cur_train_acc, cur_train_loss = eval_on_entire_dataset(sess, model, train_x, train_y,
                    args.large_batch_size, tb_prefix_and_iter, tb_writer)

    # eval on entire test/val set
    tb_prefix_and_iter = ('eval_test', iterations) if tb_writer else (None, None)
    cur_test_acc, cur_test_loss = eval_on_entire_dataset(sess, model, test_x, test_y,
                    args.test_batch_size, tb_prefix_and_iter, tb_writer)
    
    print(('{}: train acc = {:.4f}, test acc = {:.4f}, '
        + 'train loss = {:.4f}, test loss = {:.4f} ({:.2f} s)').format(iterations,
        cur_train_acc, cur_test_acc, cur_train_loss, cur_test_loss, time.time() - timerstart))

    if 'mask' in args.arch:
        percs, ones_all, size_all = [], 0, 0
        for layer in model.trainable_weights:
            assert 'mask' in layer.name, "Should be just training masks"
            #if 'bias' in layer.name:
            #    #print('bias values: ', layer.eval())
            #    continue
            mprobs = tf.stop_gradient(tf.nn.sigmoid(layer)).eval()
            num_ones = mprobs.sum() # expected value
            # old, wrong
            #nparr = layer.eval() # before sigmoid
            #num_ones = (nparr > 0).sum() + 0.5 * (nparr == 0).sum() # expected value
            #percs.append(num_ones / nparr.size)
            percs.append(num_ones / mprobs.size)
            ones_all += num_ones
            size_all += mprobs.size
        print('[Est] percent of 1s in mask (per layer):', percs)
        print('[Est] percent of 1s in mask (total):', ones_all/size_all)
        if args.dynamic_scaling:
            layer_ones = [layer.ones_in_mask for layer in list(model.layers) if 
                    'conv2D' in layer.name or 'fc' in layer.name]
            layer_mults = [layer.multiplier for layer in list(model.layers) if 
                    'conv2D' in layer.name or 'fc' in layer.name]
            layer_sizes = [tf.size(layer.kernel).eval() for layer in list(model.layers) if 
                    'conv2D' in layer.name or 'fc' in layer.name]
            l_ones = sess.run(layer_ones, feed_dict={learning_phase(): 0}) 
            l_mults = sess.run(layer_mults, feed_dict={learning_phase(): 0}) 
            print('[Act] percent of 1s in mask (per layer):', (np.array(l_ones) / np.array(layer_sizes)).tolist())
            print('[Act] percent of 1s in mask (total):', np.sum(l_ones) / np.sum(layer_sizes))
            print('layer signed constant multipliers:', l_mults)
            
    return cur_train_acc, cur_test_acc, cur_train_loss, cur_test_loss


def train_and_eval(sess, model, train_x, train_y, test_x, test_y, tb_writer, dsets, args):
    # constants
    num_batches = int(train_y.shape[0] / args.train_batch_size)
    print('Training batch size {}, number of iterations: {} per epoch, {} total'.format(
        args.train_batch_size, num_batches, args.num_epochs*num_batches))
    #dim_sum = sum([tf.size(var).eval() for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)])

    # adaptive learning schedule
    curr_lr = args.lr
    decay_epochs = [int(ep) for ep in args.decay_schedule.split(',')]
    if decay_epochs[-1] > 0:
        decay_epochs.append(-1) # need to end with something small to stop the decay
    decay_count = 0

    # initializations
    tb_summaries = tf.summary.merge(tf.get_collection('tb_train_step'))
    shuffled_indices = np.arange(train_y.shape[0]) # for no shuffling
    iterations = 0
    chunks_written = 0 # for args.save_every batches
    timerstart = time.time()

    for epoch in range(args.num_epochs):
        print('-' * 100)
        print('epoch {}  current lr {:.3g}'.format(epoch, curr_lr))
        if not args.no_shuffle:
            shuffled_indices = np.random.permutation(train_y.shape[0]) # for shuffled mini-batches

        if epoch == decay_epochs[decay_count]:
            curr_lr *= 0.1
            decay_count += 1

        for i in range(num_batches):
            # store current weights and gradients
            if args.save_weights and iterations % args.save_every == 0:
                dsets['all_weights'][chunks_written] = flatten_all(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
                chunks_written += 1

            # less frequent, larger evals
            if iterations % args.eval_every == 0:
                args.verbose = True if epoch <3 else False
                eval(sess, model, train_x, train_y, test_x, test_y, args, tb_writer, iterations)
                        
            
                if args.signed_constant and iterations < args.print_every * 3: # validate 3 times
                    print('Sanity check: signed constant values')
                    if args.signed_constant_multiplier:
                        print('Note: signed constant multiplier is {}'.format(args.signed_constant_multiplier))
                    if args.dynamic_scaling:
                        print('Note: dynamic signed constant multiplier')
                    for layer in list(model.layers):
                        if 'conv2D' in layer.name or 'fc' in layer.name:
                            #signed_kernel = layer.signed_kernel.eval()
                            signed_kernel = sess.run(layer.kernel, feed_dict={learning_phase(): 0}) 
                            print('Layer {} signed kernel shape {}, has unique values {}'.format(
                                layer.name, signed_kernel.shape, np.unique(signed_kernel).tolist()))

            # current slice for input data
            batch_indices = shuffled_indices[args.train_batch_size * i : args.train_batch_size * (i + 1)]

            # training
            fetch_dict = {'train_step': model.train_step}
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

    # save final weight values
    if args.save_weights:
        dsets['all_weights'][chunks_written] = flatten_all(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

    # save final evals
    if iterations % args.eval_every == 0:
        
        eval(sess, model, train_x, train_y, test_x, test_y, args, tb_writer, iterations)

    if 'mask' in args.arch:
        # for supermask: eval 10 times from different random bernoullies
        testaccs = []
        testlosses = []
        for sample in range(10):

            cur_test_acc, cur_test_loss = eval_on_entire_dataset(sess, model, test_x, test_y,
                                args.test_batch_size, ('eval_test', iterations), tb_writer)

            testaccs.append(cur_test_acc)
            testlosses.append(cur_test_loss)
        print("all test accs:", testaccs)
        print("all test losses:", testlosses)
        print('final test acc = {:.5f}, test loss = {:.5f}'.format(np.mean(testaccs), np.mean(testlosses)))

        percs, ones_all, size_all = [], 0, 0
        for layer in model.trainable_weights:
            mprobs = tf.stop_gradient(tf.nn.sigmoid(layer)).eval()
            num_ones = mprobs.sum() # expected value
            percs.append(num_ones / mprobs.size)
            ones_all += num_ones
            size_all += mprobs.size
            #nparr = layer.eval()
            #num_ones = (nparr > 0).sum() + 0.5 * (nparr == 0).sum() # expected value
            #percs.append(num_ones / nparr.size)
            #ones_all += num_ones
            #size_all += nparr.size
        print('[Est] percent of 1s in mask (per layer):', percs)
        print('[Est] percent of 1s in mask (total):', ones_all/size_all)
    
    if args.signed_constant: # validate in the end
        print('Sanity check: signed constant values')
        if args.dynamic_scaling:
            print('Note: dynamic signed constant multiplier')
        elif args.signed_constant_multiplier:
            print('Note: signed constant multiplier is {}'.format(args.signed_constant_multiplier))
        for layer in list(model.layers):
            if 'conv2D' in layer.name or 'fc' in layer.name:
                #signed_kernel = layer.signed_kernel.eval()
                signed_kernel = sess.run(layer.kernel, feed_dict={learning_phase(): 0}) 
                print('Layer {} signed kernel shape {}, has unique values {}'.format(
                    layer.name, signed_kernel.shape, np.unique(signed_kernel).tolist()))



def main():
    parser = make_parser()
    args = parser.parse_args()

    if args.tf_seed != -1:
        tf.random.set_random_seed(args.tf_seed)
    if not args.no_shuffle and args.shuffle_seed != -1:
        np.random.seed(args.shuffle_seed)

    # load data
    train_x, train_y = read_input_data(args.train_h5)
    test_x, test_y = read_input_data(args.test_h5) # used as val for now

    images_scale = np.max(train_x)
    if images_scale > 1:
        print('Normalizing images by a factor of {}'.format(images_scale))
        train_x = train_x / images_scale
        test_x = test_x / images_scale

    if args.test_batch_size == 0:
        args.test_batch_size = test_y.shape[0]

    print('Data shapes:', train_x.shape, train_y.shape, test_x.shape, test_y.shape)
    if train_y.shape[0] % args.train_batch_size != 0:
        print("WARNING batch size doesn't divide train set evenly")
    if train_y.shape[0] % args.large_batch_size != 0:
        print("WARNING large batch size doesn't divide train set evenly")
    if test_y.shape[0] % args.test_batch_size != 0:
        print("WARNING batch size doesn't divide test set evenly")

    # build model, masked networks
    if args.arch == 'fc_mask':
        model = masked_networks.build_fc_supermask(args)
    elif args.arch == 'conv2_mask':
        model = masked_networks.build_conv2_supermask(args)
    elif args.arch == 'conv4_mask':
        model = masked_networks.build_conv4_supermask(args)
    elif args.arch == 'conv6_mask':
        model = masked_networks.build_conv6_supermask(args)
    else:
        raise Error("Unknown architeciture {}".format(args.arch))

    init_model(model, args)
    define_training(model, args)


    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    if args.init_weights_h5:
        load_initial_weights(sess, model, args)

    for collection in ['tb_train_step']: # 'eval_train' and 'eval_test' added manually later
        tf.summary.scalar(collection + '_acc', model.accuracy, collections=[collection])
        tf.summary.scalar(collection + '_loss', model.loss, collections=[collection])

    tb_writer, hf = None, None
    dsets = {}
    if args.output_dir:
        tb_writer = tf.summary.FileWriter(args.output_dir, sess.graph)
        # set up output for gradients/weights
        if args.save_weights:
            dim_sum = sum([tf.size(var).eval() for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)])
            total_iters = args.num_epochs * int(train_y.shape[0] / args.train_batch_size)
            total_chunks = int(total_iters / args.save_every)
            hf = h5py.File(args.output_dir + '/weights', 'w-')

            # write metadata
            var_shapes = np.string_(';'.join([str(var.get_shape()) for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]))
            hf.attrs['var_shapes'] = var_shapes
            var_names = np.string_(';'.join([str(var.name) for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]))
            hf.attrs['var_names'] = var_names

            # all individual weights at every iteration, where all_weights[i] = weights before iteration i:
            dsets['all_weights'] = hf.create_dataset('all_weights', (total_chunks + 1, dim_sum), dtype='f8', compression='gzip')

    ########## Run main thing ##########
    if args.resume:
        print('='*40, 'Resuming from ckpt', '='*40)
        args.train_x, args.train_y, args.test_x, args.test_y = train_x, train_y, test_x, test_y
        args.load = args.resume
        load_ckpt_weights(sess, model, args)
    
    print('=' * 100)
    train_and_eval(sess, model, train_x, train_y, test_x, test_y, tb_writer, dsets, args)

    if tb_writer:
        tb_writer.close()
    if hf:
        hf.close()

if __name__ == '__main__':
    main()

