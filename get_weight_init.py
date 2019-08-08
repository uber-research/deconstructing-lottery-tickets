
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

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default = 0)
    parser.add_argument('--weights_h5', type=str, required=True)
    parser.add_argument('--output_h5', type=str, required=True)
    parser.add_argument('--method', type=str, required=True, choices = ('large_final', 'small_final', 'random', 'large_init', 'small_init', 'large_init_large_final', 'small_init_small_final', 'magnitude_increase', 'movement','snip','grad', 'large_final_diff_sign', 'large_final_same_sign', 'magnitude_increase_diff_sign', 'magnitude_increase_same_sign', 'large_final_diff_sign_single', 'large_final_same_sign_single'))
    parser.add_argument('--prune_base', type=str, required=True, help = 'comma-separated list')
    parser.add_argument('--prune_power', type=int, required=True)
    parser.add_argument('--layer_cutoff', type=str, required=True, help = 'comma-separated list')
    parser.add_argument('--final_weight_interpolation', type=float, default = 1)
    parser.add_argument('--final_weights_ind', type=int, default = -1)
    return parser

def get_mask_by_layer_by_weight(weight_one_layer, percentile, cur_mask = None):    
    layer_shape = weight_one_layer.shape
    weight_one_layer = weight_one_layer.flatten()

    if cur_mask is not None:
        cur_mask = cur_mask.flatten()
        #if something is already masked out, set it to a high number to avoid pruning again
        weight_one_layer[cur_mask == 0] = 9999
    else:
        cur_mask = np.ones(weight_one_layer.shape)
    
    prune_num = np.int(np.sum(cur_mask) * percentile / 100)
    weight_ind = np.lexsort((np.random.random(weight_one_layer.size), weight_one_layer))[:prune_num]
    
    cur_mask[weight_ind] = 0
    
    return cur_mask.reshape(layer_shape)

# number of parameters you pruned if you prune at prune_perc for both weight1 and weight2
def pruned_count(weight1, weight2, prune_perc):
    p1 = np.percentile(weight1, prune_perc)
    p2 = np.percentile(weight2, prune_perc)
    return weight1.size - ((weight1 >= p1) * (weight2 >= p2)).sum()

def bsearch_quadrant_threshold(weight1, weight2, target_percentile, eps=1e-4):
    min_percentile, max_percentile = 0, target_percentile # lower and upper bounds to prune percentage
    target_pruned_count = int(weight1.size * target_percentile / 100) # how many do you want to prune

    cur_percentile = (min_percentile + max_percentile) / 2.0
    cur_count = pruned_count(weight1, weight2, cur_percentile)

    while cur_count != target_pruned_count:
        if max_percentile - min_percentile < eps:
            return max_percentile
        if cur_count > target_pruned_count: # pruned too many, lower prune percentile
            max_percentile = cur_percentile
        else: # pruned not enough, increase prune percentile
            min_percentile = cur_percentile
        cur_percentile = (min_percentile + max_percentile) / 2.0
        cur_count = pruned_count(weight1, weight2, cur_percentile)

    return cur_percentile

def split_and_shape(one_time_slice, shapes):
    variables = []
    offset = 0
    for shape in shapes:
        num_params = np.prod(shape)
        variables.append(one_time_slice[offset : offset + num_params].reshape(shape))
        offset += num_params
    return variables

def main():
    parser = make_parser()
    args = parser.parse_args()
#     np.random.seed(args.seed)
    
    prune_percentiles = []
    prune_base = [float(pb) for pb in args.prune_base.split(',')]
    layer_cutoff = [int(lc) for lc in args.layer_cutoff.split(',')]
    
    i = 0
    for k, lc in enumerate(layer_cutoff):
        while i < lc:
            prune_percentiles.append((1 - (prune_base[k] ** args.prune_power)) * 100.0)
            prune_percentiles.append(0) #hacky way to not mask bias for now
            i += 2
            
    print('==prune percentiles are: ')
    print(prune_percentiles)
    
    hf = h5py.File(args.weights_h5, 'r')
    raw_initial_weights = np.array(hf.get('all_weights')[0])
    raw_final_weights = np.array(hf.get('all_weights')[args.final_weights_ind])
    init_grads = np.squeeze(np.array(hf.get('one_iter_grads')))

    raw_shapes = hf.attrs['var_shapes']
    current_mask = np.array(hf.get('mask_values'))
    shapes = [literal_eval(s) for s in raw_shapes.decode('utf-8').split(';')]
    hf.close()

    initial_weights = raw_initial_weights
    initial_weights = split_and_shape(initial_weights, shapes)
    
    #create a dummy mask if no mask exists:
    if current_mask.size == 1:
        print('==no mask exists - generating first mask')
        current_mask = np.ones(raw_initial_weights.shape)
        
    current_mask = split_and_shape(current_mask, shapes)
    mask_values = initial_weights[:]
    num_layers = len(initial_weights)
    
    if args.method == 'large_final':
        cumulative_helped = np.abs(raw_final_weights)
        cumulative_helped = split_and_shape(cumulative_helped, shapes)
        
        for i, pp in enumerate(prune_percentiles):
            if i % 2 == 1:
                mask_values[i] = current_mask[i]
            else:
                mask_values[i] = get_mask_by_layer_by_weight(cumulative_helped[i], pp, current_mask[i])
                
    elif args.method == 'large_final_diff_sign':
        init_sign = np.sign(raw_initial_weights)
        final_sign = np.sign(raw_final_weights)
        same_sign_ind = np.squeeze(np.where(init_sign == final_sign))
        cumulative_helped = np.abs(raw_final_weights)
        cumulative_helped[same_sign_ind] = 0
        cumulative_helped = split_and_shape(cumulative_helped, shapes)
        
        for i, pp in enumerate(prune_percentiles):
            if i % 2 == 1:
                mask_values[i] = current_mask[i]
            else:
                mask_values[i] = get_mask_by_layer_by_weight(cumulative_helped[i], pp, current_mask[i])
                
    elif args.method == 'large_final_same_sign':
        init_sign = np.sign(raw_initial_weights)
        final_sign = np.sign(raw_final_weights)
        diff_sign_ind = np.squeeze(np.where(init_sign != final_sign))
        cumulative_helped = np.abs(raw_final_weights)
        cumulative_helped[diff_sign_ind] = 0
        cumulative_helped = split_and_shape(cumulative_helped, shapes)
        
        for i, pp in enumerate(prune_percentiles):
            if i % 2 == 1:
                mask_values[i] = current_mask[i]
            else:
                mask_values[i] = get_mask_by_layer_by_weight(cumulative_helped[i], pp, current_mask[i])
            
    elif args.method == 'large_final_diff_sign_single':
        init_sign = np.sign(raw_initial_weights)
        final_sign = np.sign(raw_final_weights)
        sign_ind = np.squeeze(np.where(init_sign != final_sign))
        mask_values = np.zeros(raw_initial_weights.shape)
        mask_values[sign_ind] = 1
        mask_values = split_and_shape(mask_values, shapes)
        
                
    elif args.method == 'large_final_same_sign_single':
        init_sign = np.sign(raw_initial_weights)
        final_sign = np.sign(raw_final_weights)
        sign_ind = np.squeeze(np.where(init_sign == final_sign))
        mask_values = np.zeros(raw_initial_weights.shape)
        mask_values[sign_ind] = 1
        mask_values = split_and_shape(mask_values, shapes)
        
                
    elif args.method == 'small_final':
        cumulative_helped = -np.abs(raw_final_weights)
        cumulative_helped = split_and_shape(cumulative_helped, shapes)
        
        for i, pp in enumerate(prune_percentiles):
            if i % 2 == 1:
                mask_values[i] = current_mask[i]
            else:
                mask_values[i] = get_mask_by_layer_by_weight(cumulative_helped[i], pp, current_mask[i])
            
    elif args.method == 'random':
        cumulative_helped = np.arange(len(raw_initial_weights)) / len(raw_initial_weights)
        shuffle(cumulative_helped)
        cumulative_helped = split_and_shape(cumulative_helped, shapes)
        
        for i, pp in enumerate(prune_percentiles):
            if i % 2 == 1:
                mask_values[i] = current_mask[i]
            else:
                mask_values[i] = get_mask_by_layer_by_weight(cumulative_helped[i], pp, current_mask[i])


    elif args.method == 'large_init':
        cumulative_helped = np.abs(raw_initial_weights)
        cumulative_helped = split_and_shape(cumulative_helped, shapes)
        
        for i, pp in enumerate(prune_percentiles):
            if i % 2 == 1:
                mask_values[i] = current_mask[i]
            else:
                mask_values[i] = get_mask_by_layer_by_weight(cumulative_helped[i], pp, current_mask[i])
            
    elif args.method == 'small_init':
        cumulative_helped = -np.abs(raw_initial_weights)
        cumulative_helped = split_and_shape(cumulative_helped, shapes)
        
        for i, pp in enumerate(prune_percentiles):
            if i % 2 == 1:
                mask_values[i] = current_mask[i]
            else:
                mask_values[i] = get_mask_by_layer_by_weight(cumulative_helped[i], pp, current_mask[i])
            
            
    elif args.method == 'large_init_large_final':
        cumulative_helped_i = np.abs(raw_initial_weights)
        cumulative_helped_i = split_and_shape(cumulative_helped_i, shapes)
        cumulative_helped_f = np.abs(raw_final_weights)
        cumulative_helped_f = split_and_shape(cumulative_helped_f, shapes)
        mask_values_i = mask_values[:]
        mask_values_f = mask_values[:]
        
        for i, pp in enumerate(prune_percentiles):
            if i % 2 == 1:
                mask_values_i[i] = current_mask[i]
                mask_values_f[i] = current_mask[i]
            else:
                cumulative_helped_i[i][~current_mask[i].astype('bool')] = -9999
                cumulative_helped_f[i][~current_mask[i].astype('bool')] = -9999
                existing_mask = np.sum(current_mask[i])
                total_size = current_mask[i].flatten().size
                new_pp = ((total_size - existing_mask) + existing_mask * (pp / 100)) / total_size
                new_pp = new_pp * 100
                threshold = bsearch_quadrant_threshold(cumulative_helped_i[i],
                                                    cumulative_helped_f[i],
                                                    target_percentile = new_pp)

                mask_values_i[i] = get_mask_by_layer_by_weight(cumulative_helped_i[i], threshold)
                mask_values_f[i] = get_mask_by_layer_by_weight(cumulative_helped_f[i], threshold)

        for i in range(len(mask_values)):
            mask_values[i] = mask_values_i[i] * mask_values_f[i]
            
    elif args.method == 'small_init_small_final':
        cumulative_helped_i = -np.abs(raw_initial_weights)
        cumulative_helped_i = split_and_shape(cumulative_helped_i, shapes)
        cumulative_helped_f = -np.abs(raw_final_weights)
        cumulative_helped_f = split_and_shape(cumulative_helped_f, shapes)
        mask_values_i = mask_values[:]
        mask_values_f = mask_values[:]
        
        for i, pp in enumerate(prune_percentiles):
            if i % 2 == 1:
                mask_values_i[i] = current_mask[i]
                mask_values_f[i] = current_mask[i]
            else:
                cumulative_helped_i[i][~current_mask[i].astype('bool')] = -9999
                cumulative_helped_f[i][~current_mask[i].astype('bool')] = -9999
                existing_mask = np.sum(current_mask[i])
                total_size = current_mask[i].flatten().size
                new_pp = ((total_size - existing_mask) + existing_mask * (pp / 100)) / total_size
                new_pp = new_pp * 100
                threshold = bsearch_quadrant_threshold(cumulative_helped_i[i],
                                                    cumulative_helped_f[i],
                                                    target_percentile = new_pp)

                mask_values_i[i] = get_mask_by_layer_by_weight(cumulative_helped_i[i], threshold)
                mask_values_f[i] = get_mask_by_layer_by_weight(cumulative_helped_f[i], threshold)
            
        for i in range(len(mask_values)):
            mask_values[i] = mask_values_i[i] * mask_values_f[i]
            
    elif args.method == 'magnitude_increase':
        cumulative_helped = np.abs(raw_final_weights) - np.abs(raw_initial_weights)
        cumulative_helped = split_and_shape(cumulative_helped, shapes)
        
        for i, pp in enumerate(prune_percentiles):
            if i % 2 == 1:
                mask_values[i] = current_mask[i]
            else:
                mask_values[i] = get_mask_by_layer_by_weight(cumulative_helped[i], pp, current_mask[i])
                
    elif args.method == 'magnitude_increase_diff_sign':
        init_sign = np.sign(raw_initial_weights)
        final_sign = np.sign(raw_final_weights)
        same_sign_ind = np.squeeze(np.where(init_sign == final_sign))
        cumulative_helped = np.abs(raw_final_weights) - np.abs(raw_initial_weights)
        cumulative_helped[same_sign_ind] = 0
        cumulative_helped = split_and_shape(cumulative_helped, shapes)
        
        for i, pp in enumerate(prune_percentiles):
            if i % 2 == 1:
                mask_values[i] = current_mask[i]
            else:
                mask_values[i] = get_mask_by_layer_by_weight(cumulative_helped[i], pp, current_mask[i])
                
    elif args.method == 'magnitude_increase_same_sign':
        init_sign = np.sign(raw_initial_weights)
        final_sign = np.sign(raw_final_weights)
        diff_sign_ind = np.squeeze(np.where(init_sign != final_sign))
        cumulative_helped = np.abs(raw_final_weights) - np.abs(raw_initial_weights)
        cumulative_helped[diff_sign_ind] = 0
        cumulative_helped = split_and_shape(cumulative_helped, shapes)
        
        for i, pp in enumerate(prune_percentiles):
            if i % 2 == 1:
                mask_values[i] = current_mask[i]
            else:
                mask_values[i] = get_mask_by_layer_by_weight(cumulative_helped[i], pp, current_mask[i])
            
    
    elif args.method == 'movement':
        cumulative_helped = np.abs(raw_final_weights - raw_initial_weights)
        cumulative_helped = split_and_shape(cumulative_helped, shapes)
        
        for i, pp in enumerate(prune_percentiles):
            if i % 2 == 1:
                mask_values[i] = current_mask[i]
            else:
                mask_values[i] = get_mask_by_layer_by_weight(cumulative_helped[i], pp, current_mask[i])
            

    elif args.method == 'snip':
        
        cumulative_helped = np.abs(init_grads * raw_initial_weights)
        cumulative_helped = split_and_shape(cumulative_helped, shapes)
        
        for i, pp in enumerate(prune_percentiles):
            if i % 2 == 1:
                mask_values[i] = current_mask[i]
            else:
                mask_values[i] = get_mask_by_layer_by_weight(cumulative_helped[i], pp, current_mask[i])
            
    elif args.method == 'grad':
        cumulative_helped = np.abs(init_grads)
        cumulative_helped = split_and_shape(cumulative_helped, shapes)
        
        for i, pp in enumerate(prune_percentiles):
            if i % 2 == 1:
                mask_values[i] = current_mask[i]
            else:
                mask_values[i] = get_mask_by_layer_by_weight(cumulative_helped[i], pp, current_mask[i])
            
        
    mask_values = np.concatenate([x.flatten() for x in mask_values])
#     current_mask = np.concatenate([x.flatten() for x in current_mask])
    train_diff = raw_final_weights - raw_initial_weights
    final_weights = raw_initial_weights + train_diff * args.final_weight_interpolation

    hf = h5py.File(args.output_h5, 'w')
    hf.attrs['prune_base'] = args.prune_base
    hf.attrs['prune_power'] = args.prune_power
    hf.attrs['var_shapes'] = raw_shapes
    hf.attrs['prune_percentiles'] = prune_percentiles
    hf.create_dataset('initial_weights', data=raw_initial_weights)
    hf.create_dataset('final_weights', data=final_weights)
    hf.create_dataset('mask_values', data=mask_values)
#     hf.create_dataset('previous_mask', data=current_mask)
    hf.close()

if __name__ == '__main__':
    main()

