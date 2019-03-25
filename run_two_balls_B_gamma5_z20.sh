#! /bin/sh

python main.py --dataset cube_small --seed 1 --lr 1e-3 --beta1 0.9 --beta2 0.999 \
--objective B --model B --batch_size 64 --z_dim 20 --max_iter 1.6e4 \
--viz_name two_balls_B_gamma5_z20 --num_workers 4 --gamma 5 \
--image_size 24 --viz_on False