#! /bin/sh

python main.py --dataset two_balls --seed 1 --lr 1e-5 --beta1 0.9 --beta2 0.999 \
--objective B --model B --batch_size 1000 --z_dim 10 --max_iter 1.6e5 \
--viz_name adadelta_two_balls_gamma01 --num_workers 4 --gamma 0.1 --viz_on False