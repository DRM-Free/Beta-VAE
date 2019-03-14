#! /bin/sh

python main.py --dataset cube_small --seed 1 --lr 1e-3 --beta1 0.9 --beta2 0.999 \
--objective H --model H --batch_size 64 --z_dim 3 --max_iter 1.5e4 \
--viz_name cube_small_H_beta20_z3 --num_workers 4 --beta 20