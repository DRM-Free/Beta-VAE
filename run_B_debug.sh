#! /bin/sh

python main.py --dataset cube_small --seed 1 --lr 1e-5 --beta1 0.9 --beta2 0.999 \
--objective B --model B --batch_size 500 --z_dim 3 --max_iter 100 \
--save_step 20 --viz_name debug_test_4 --num_workers 4 --gamma 0.1 --viz_on False