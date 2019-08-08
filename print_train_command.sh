# Copyright (c) 2019 Uber Technologies, Inc.

# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project. 

# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash
type=$1
network=$2
orig_dir_name=$3
seed=$4
execute=$5

case "$network" in
        fc)
                case "$type" in 
                        iter)
                                filedir=iter_lot_fc_orig
                                ;;
                        one)
                                filedir=lot_fc_orig
                esac
                num_epochs=55
                arch=fc_lot
                lr=0.0012
                data=mnist
                large_batch_size=5000
                small_batch_size=0
                ;;
        conv2)
                case "$type" in 
                        iter)
                                filedir=iter_lot_conv2_orig
                                ;;
                        one)
                                filedir=lot_conv2_orig
                esac
                num_epochs=27
                arch=conv2_lot
                lr=0.0002
                data=cifar10
                large_batch_size=2000
                small_batch_size=2000
                ;;

        conv4)
                case "$type" in 
                        iter)
                                filedir=iter_lot_conv4_orig
                                ;;
                        one)
                                filedir=lot_conv4_orig
                esac
                num_epochs=34
                arch=conv4_lot
                lr=0.0003
                data=cifar10
                large_batch_size=2000
                small_batch_size=2000
                ;;

        conv6)
                case "$type" in 
                        iter)
                                filedir=iter_lot_conv6_orig
                                ;;
                        one)
                                filedir=lot_conv6_orig
                esac
                num_epochs=40
                arch=conv6_lot
                lr=0.0003
                data=cifar10
                large_batch_size=2000
                small_batch_size=2000
        ;;
    *)
        echo error!!!!
esac

case "$execute" in
    t)
        resman -d "./results/${filedir}/" -r "${orig_dir_name}_seed_${seed}" -t "{runname}" -- python train.py --train_h5 "./data/${data}_train" --test_h5 "./data/${data}_test" --val_h5 "./data/${data}_val" --train_batch_size 60 --num_epochs $num_epochs --eval_every 100 --print_every 100 --save_weights --save_loss --arch $arch --seed $seed --opt adam --lr $lr --mode save_all --large_batch_size $large_batch_size --test_batch_size $small_batch_size --val_batch_size $small_batch_size
        ;;
    *)
        echo resman -d "./results/${filedir}/" -r "${orig_dir_name}_seed_${seed}" -t "{runname}" -- python train.py --train_h5 "./data/${data}_train" --test_h5 "./data/${data}_test" --val_h5 "./data/${data}_val" --train_batch_size 60 --num_epochs $num_epochs --eval_every 100 --print_every 100 --save_weights --save_loss --arch $arch --seed $seed --opt adam --lr $lr --mode save_all --large_batch_size $large_batch_size --test_batch_size $small_batch_size --val_batch_size $small_batch_size
esac
