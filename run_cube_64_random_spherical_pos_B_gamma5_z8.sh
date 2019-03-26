#! /bin/sh

python main.py --dataset cube_small --seed 1 --lr 1e-5 --beta1 0.9 --beta2 0.999 \
--objective B --model B --batch_size 64 --z_dim 8 --max_iter 1.6e5 \
--viz_name cube_64_random_spherical_pos_B_gamma5_z8 --num_workers 4 --gamma 2 --viz_on False