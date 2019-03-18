import argparse
import numpy as np
import torch
from solver import Solver
from utils import str2bool


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def main(args):
    net = Solver(args)

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='interactive latent space explorer')
    parser.add_argument('--load_directory', default='checkpoints/cube_small_B_gamma5_z3',
                        type=str, help='directory containing the checkpoint to be loaded')
    parser.add_argument('--viz_on', default=True,
                        type=str2bool, help='enable visdom visualization')
    parser.add_argument('--viz_name', default='main',
                        type=str, help='visdom env name')
    parser.add_argument('--viz_port', default=8097,
                        type=str, help='visdom port number')

    parser.add_argument('--z_dim', default=10, type=int,
                        help='dimension of the representation z')
    parser.add_argument('--beta', default=4, type=float,
                        help='beta parameter for KL-term in original beta-VAE')
    parser.add_argument('--objective', default='H', type=str,
                        help='beta-vae objective proposed in Higgins et al. or Burgess et al. H/B')
    parser.add_argument('--model', default='H', type=str,
                        help='model proposed in Higgins et al. or Burgess et al. H/B')
    parser.add_argument('--gamma', default=1000, type=float,
                        help='gamma parameter for KL-term in understanding beta-VAE')
    parser.add_argument('--C_max', default=25, type=float,
                        help='capacity parameter(C) of bottleneck channel')
    parser.add_argument('--C_stop_iter', default=1e5, type=float,
                        help='when to stop increasing the capacity')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--beta1', default=0.9, type=float,
                        help='Adam optimizer beta1')
    parser.add_argument('--beta2', default=0.999,
                        type=float, help='Adam optimizer beta2')

    parser.add_argument('--dset_dir', default='data',
                        type=str, help='dataset directory')
    parser.add_argument('--dataset', default='cube_small',
                        type=str, help='dataset name')
    parser.add_argument('--image_size', default=64, type=int,
                        help='image size. now only (64,64) is supported')
