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
    #from get_weight_init
    parser.add_argument('--output_dir', type=str, required = True)
    parser.add_argument('--train_h5', type=str, required=True)
    parser.add_argument('--test_h5', type=str, required=True)
    parser.add_argument('--weights_choice', type=str, default = 'lot', choices = ('lot', 'final')) 
    parser.add_argument('--signed_constant', type=float)
    parser.add_argument('--init_suffix', type=str, default = '_init_weights') 
    parser.add_argument('--weights_name', type=str, default = 'weights') 
    
    #from train
    parser.add_argument('--seed', type=int, default = 0) #used to be int
    parser.add_argument('--arch', type=str, default='fc_lot', choices=('fc_lot', 'conv2_lot', 'conv4_lot', 'conv6_lot'), help='network architecture')
    parser.add_argument('--opt', type=str, default='sgd', choices=('sgd', 'rmsprop', 'adam'))
    parser.add_argument('--lr', type=float, default=.01, help='suggested: .01 sgd, .001 rmsprop, .0001 adam')
    parser.add_argument('--mom', type=float, default=.9, help='momentum (only has effect for sgd/rmsprop)')
    parser.add_argument('--l2', type=float, default=0)
    parser.add_argument('--large_batch_size', type=int, default=11000, help='mnist: 11000, cifar: 5000') 
    parser.add_argument('--test_batch_size', type=int, default=0) # do 0 for all 

    #mask scenarios
    parser.add_argument('--mask_scenarios', type=str, default = 'large_final,small_final,random,large_init,small_init,large_init_large_final,small_init_small_final,magnitude_increase,movement,snip,grad,large_final_diff_sign,large_final_same_sign,magnitude_increase_diff_sign,magnitude_increase_same_sign')
    parser.add_argument('--prune_base', type=str, required = True, help = 'comma-separated list')
    parser.add_argument('--prune_power', type=str, required = True, help = 'comma-separated list')
    parser.add_argument('--layer_cutoff', type=str, required = True, help = 'comma-separated list')
    
    #train lottery with masks
    parser.add_argument('--exp', type=str, default='none', choices = ('none', 'random_reinit', 'random_reshuffle', 'signed_reinit', 'signed_reshuffle', 'signed_constant', 'rand_signed_constant'), help='experiment scenarios')    
    parser.add_argument('--signed_constant_scaling', type=float, default = 1)
    parser.add_argument('--output_name_prefix', type=str, default='all_init_accuracy_', help='prefix name for accuracy data')
    parser.add_argument('--final_weights_ind', type=str, default = '-1')
    parser.add_argument('--final_weight_interpolation', type=str, default = '1')
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
def eval_on_entire_dataset(sess, model, input_x, input_y, batch_size):
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

    return acc, loss_no_reg

#################

def split_and_shape(one_time_slice, shapes):
    variables = []
    offset = 0
    for shape in shapes:
        num_params = np.prod(shape)
        variables.append(one_time_slice[offset : offset + num_params].reshape(shape))
        offset += num_params
    return variables

def train_and_eval(sess, model, train_x, train_y, test_x, test_y, dsets, args, current_run):

    cur_train_acc, cur_train_loss = eval_on_entire_dataset(sess, model, train_x, train_y, args.large_batch_size)
    # eval on entire test/val set
    cur_test_acc, cur_test_loss = eval_on_entire_dataset(sess, model, test_x, test_y, args.test_batch_size)
    
    dsets['train_accuracy'][current_run] = cur_train_acc
    dsets['train_loss'][current_run] = cur_train_loss
    dsets['test_accuracy'][current_run] = cur_test_acc
    dsets['test_loss'][current_run] = cur_test_loss

def main():
    parser = make_parser()
    args = parser.parse_args()
    
    mask_scenarios = [mask for mask in args.mask_scenarios.split(',')]
    
    pp = [p for p in args.prune_power.split(',')]
    
    scenario_names = []
    pp_names = []
    total_run = len(pp) * len(mask_scenarios)
    current_run = 0
    
    output_directory = args.output_dir + '/mask_scenarios/'
    if not os.path.exists(output_directory):
        print('directory does not exist, creating : '+ output_directory)
        os.makedirs(output_directory)
    
    hf_output = h5py.File(output_directory + args.output_name_prefix + args.weights_choice + '_exp_' + args.exp + '_fw_ind_' + args.final_weights_ind, 'w-')
    
    dsets = {}
    dsets['train_accuracy'] = hf_output.create_dataset('train_accuracy', (total_run, 1), dtype='f8')
    dsets['train_loss'] = hf_output.create_dataset('train_loss', (total_run, 1), dtype='f8')
    dsets['test_accuracy'] = hf_output.create_dataset('test_accuracy', (total_run, 1), dtype='f8')
    dsets['test_loss'] = hf_output.create_dataset('test_loss', (total_run, 1), dtype='f8')

    
    for p in range(len(pp)):
        print('=====running percentile power ' + pp[p])
        for mask in mask_scenarios:
            print('=====running mask scenario ' + mask)
            scenario_names.append(mask)
            pp_names.append(p)

            init_weight_name = output_directory + 'pp_' + pp[p] + '_' + mask + '_' + args.exp + '_' + str(args.final_weights_ind) + args.init_suffix
            if os.path.isfile(init_weight_name) == False:
                os.system('python get_weight_init.py --method ' + mask + ' --weights_h5 ' + args.output_dir + '/' + args.weights_name + ' --output_h5 ' + init_weight_name + ' --prune_base ' + args.prune_base + ' --prune_power ' + str(pp[p]) + ' --layer_cutoff ' + args.layer_cutoff + ' --seed ' + str(args.seed) + ' --final_weight_interpolation ' + args.final_weight_interpolation + ' --final_weights_ind ' + args.final_weights_ind)
                
            print('starting to calc accuracy')
            #get accuracy===============================================================
            np.random.seed(args.seed)
            tf.set_random_seed(args.seed)

            #load data
            train_x, train_y = read_input_data(args.train_h5)
            test_x, test_y = read_input_data(args.test_h5)

            if 'mnist' in args.train_h5:
                input_dim = '28,28,1'
            elif 'cifar10' in args.train_h5:
                input_dim = '32,32,3'
        
            hf = h5py.File(init_weight_name, 'r')
            shapes = [literal_eval(s) for s in hf.attrs['var_shapes'].decode('utf-8').split(';')]
            
            initial_weights = np.array(hf.get('initial_weights'), dtype='f16')
            initial_weights = split_and_shape(initial_weights, shapes)
                
            final_weights = np.array(hf.get('final_weights'), dtype='f16')
            final_weights = split_and_shape(final_weights, shapes)
                
            mask_values = np.array(hf.get('mask_values'), dtype='f16')
            mask_values = split_and_shape(mask_values, shapes)
            
            hf.close()
            
            images_scale = np.max(train_x)
            if images_scale > 1:
                print('Normalizing images by a factor of {}'.format(images_scale))
                train_x = train_x / images_scale
                test_x = test_x / images_scale

            if args.test_batch_size == 0:
                args.test_batch_size = test_y.shape[0]

            print('Data shapes:', train_x.shape, train_y.shape, test_x.shape, test_y.shape)
            if train_y.shape[0] % args.large_batch_size != 0:
                print("WARNING large batch size doesn't divide train set evenly")
            if test_y.shape[0] % args.test_batch_size != 0:
                print("WARNING batch size doesn't divide test set evenly")

            #build model 
            if args.weights_choice == 'lot':
                if args.arch == 'fc_lot':
                    model = network_builders.build_masked_fc_lottery(args, mask_values)
                elif args.arch == 'conv2_lot':
                    model = network_builders.build_masked_conv2_lottery(args, mask_values)
                elif args.arch == 'conv4_lot':
                    model = network_builders.build_masked_conv4_lottery(args, mask_values)
                elif args.arch == 'conv6_lot':
                    model = network_builders.build_masked_conv6_lottery(args, mask_values)

            elif args.weights_choice == 'final':
                if args.arch == 'fc_lot':
                    model = network_builders.build_frozen_fc_lottery(args, final_weights, mask_values)
                elif args.arch == 'conv2_lot':
                    model = network_builders.build_frozen_conv2_lottery(args, final_weights, mask_values)
                elif args.arch == 'conv4_lot':
                    model = network_builders.build_frozen_conv4_lottery(args, final_weights, mask_values)
                elif args.arch == 'conv6_lot':
                    model = network_builders.build_frozen_conv6_lottery(args, final_weights, mask_values)

                    
            init_model(model, input_dim)
            define_training(model, args)

            sess = tf.InteractiveSession()
            sess.run(tf.global_variables_initializer())
            
            signed_constants = []
            
            if args.exp == 'none':
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
            
            train_and_eval(sess, model, train_x, train_y, test_x, test_y, dsets, args, current_run)
            current_run += 1
            sess.close()
            tf.reset_default_graph()
            
    hf_output.attrs['scenario_names'] = scenario_names
    hf_output.attrs['pp_names'] = pp_names
    dsets['signed_constants'] = hf_output.create_dataset('signed_constants', data=np.array(signed_constants))
    hf_output.close()

                
if __name__ == '__main__':
    main()

            
