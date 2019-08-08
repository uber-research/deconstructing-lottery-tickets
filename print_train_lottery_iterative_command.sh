# Copyright (c) 2019 Uber Technologies, Inc.

# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project. 

# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash
network=$1
orig_dir_name=$2
seed=$3
method=$4
weight_ind=$5
model_type=$6
exp=$7
execute=$8

case "$weight_ind" in
    "")
        weight_ind=-1
esac

case "$network" in
        fc)
                filedir=iter_lot_fc_orig
		num_epochs=55
                arch=fc_lot
                lr=0.0012
                data=mnist
                large_batch_size=5000
                small_batch_size=0
                prune_base=0.8,0.9
                layer_cutoff=4,6
		case "$weight_ind" in
			-1)
				save_special_iter=-1
				;;
			*)
				let "save_special_iter=$weight_ind*100"
		esac
                ;;
        conv2)
                filedir=iter_lot_conv2_orig
                num_epochs=27
                arch=conv2_lot
                lr=0.0002
                data=cifar10
                large_batch_size=1000
                small_batch_size=1000
                prune_base=0.9,0.8,0.9
                layer_cutoff=4,8,10
                case "$weight_ind" in
                        -1)
                                save_special_iter=-1
                                ;;
                        *)
                                let "save_special_iter=$weight_ind*100"
                esac                
		;;

        conv4)
                filedir=iter_lot_conv4_orig
                num_epochs=34
                arch=conv4_lot
                lr=0.0003
                data=cifar10
                large_batch_size=1000
                small_batch_size=1000
                prune_base=0.9,0.8,0.9
                layer_cutoff=8,12,14
		case "$weight_ind" in
                        -1)
                                save_special_iter=-1
                                ;;
                        *)
                                let "save_special_iter=$weight_ind*100"
                esac  
		;;

        conv6)
                filedir=iter_lot_conv6_orig
                num_epochs=40
                arch=conv6_lot
                lr=0.0003
                data=cifar10
                large_batch_size=1000
                small_batch_size=1000
                prune_base=0.85,0.8,0.9
                layer_cutoff=12,16,18
                case "$weight_ind" in
                        -1)
                                save_special_iter=-1
                                ;;
                        *)
                                let "save_special_iter=$weight_ind*100"
                esac
                ;;
           *)
                echo error!!!
esac

case "$model_type" in
        "")
                model_type=mask
esac

case "$exp" in
        "")
                exp=none
esac

for pb in {1..24}
do
    let "pb2=$pb-1"
    case "$execute" in
        t)
        case $pb in
            1)
                resman -d "./results/${filedir}/${orig_dir_name}_seed_${seed}/${method}_weight_ind${weight_ind}_${model_type}_exp_${exp}" -r "pp${pb}" -t "{runname}" -- python train_lottery.py --train_h5 "./data/${data}_train" --test_h5 "./data/${data}_test" --val_h5 "./data/${data}_val" --train_batch_size 60 --num_epochs $num_epochs --eval_every 100 --print_every 100 --save_weights --save_loss --arch $arch --seed $seed --opt adam --lr $lr --mode save_res --method $method --prune_base $prune_base --prune_power 1 --layer_cutoff $layer_cutoff --prev_weights "./results/${filedir}/${orig_dir_name}_seed_${seed}" --orig_weights "./results/${filedir}/${orig_dir_name}_seed_${seed}" --final_weights_ind $weight_ind --save_special_iter $save_special_iter --model_type $model_type --exp $exp --large_batch_size $large_batch_size --test_batch_size $small_batch_size --val_batch_size $small_batch_size
                ;;
            *)
                resman -d "./results/${filedir}/${orig_dir_name}_seed_${seed}/${method}_weight_ind${weight_ind}_${model_type}_exp_${exp}" -r "pp${pb}" -t "{runname}" -- python train_lottery.py --train_h5 "./data/${data}_train" --test_h5 "./data/${data}_test" --val_h5 "./data/${data}_val" --train_batch_size 60 --num_epochs $num_epochs --eval_every 100 --print_every 100 --save_weights --save_loss --arch $arch --seed $seed --opt adam --lr $lr --mode save_res --method $method --prune_base $prune_base --prune_power 1 --layer_cutoff $layer_cutoff --prev_weights "./results/${filedir}/${orig_dir_name}_seed_${seed}/${method}_weight_ind${weight_ind}_${model_type}_exp_${exp}/pp${pb2}" --orig_weights "./results/${filedir}/${orig_dir_name}_seed_${seed}" --final_weights_ind 2 --save_special_iter $save_special_iter --model_type $model_type --exp $exp --large_batch_size $large_batch_size --test_batch_size $small_batch_size --val_batch_size $small_batch_size

            esac
        ;;
    *)
        case $pb in
            1)
                echo resman -d "./results/${filedir}/${orig_dir_name}_seed_${seed}/${method}_weight_ind${weight_ind}_${model_type}_exp_${exp}" -r "pp${pb}" -t "{runname}" -- python train_lottery.py --train_h5 "./data/${data}_train" --test_h5 "./data/${data}_test" --val_h5 "./data/${data}_val" --train_batch_size 60 --num_epochs $num_epochs --eval_every 100 --print_every 100 --save_weights --save_loss --arch $arch --seed $seed --opt adam --lr $lr --mode save_res --method $method --prune_base $prune_base --prune_power 1 --layer_cutoff $layer_cutoff --prev_weights "./results/${filedir}/${orig_dir_name}_seed_${seed}" --orig_weights "./results/${filedir}/${orig_dir_name}_seed_${seed}" --final_weights_ind $weight_ind --save_special_iter $save_special_iter --model_type $model_type --exp $exp --large_batch_size $large_batch_size --test_batch_size $small_batch_size --val_batch_size $small_batch_size
                ;;
            *)
                echo resman -d "./results/${filedir}/${orig_dir_name}_seed_${seed}/${method}_weight_ind${weight_ind}_${model_type}_exp_${exp}" -r "pp${pb}" -t "{runname}" -- python train_lottery.py --train_h5 "./data/${data}_train" --test_h5 "./data/${data}_test" --val_h5 "./data/${data}_val" --train_batch_size 60 --num_epochs $num_epochs --eval_every 100 --print_every 100 --save_weights --save_loss --arch $arch --seed $seed --opt adam --lr $lr --mode save_res --method $method --prune_base $prune_base --prune_power 1 --layer_cutoff $layer_cutoff --prev_weights "./results/${filedir}/${orig_dir_name}_seed_${seed}/${method}_weight_ind${weight_ind}_${model_type}_exp_${exp}/pp${pb2}" --orig_weights "./results/${filedir}/${orig_dir_name}_seed_${seed}" --final_weights_ind 2 --save_special_iter $save_special_iter --model_type $model_type --exp $exp --large_batch_size $large_batch_size --test_batch_size $small_batch_size --val_batch_size $small_batch_size
        esac
esac
done
