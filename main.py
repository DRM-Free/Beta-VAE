"""main.py"""
# TODO Experiment with different experimental conditions : beta/gamma, size of the dataset, position selection, order in which images are presented...
# TODO support higher image size
# TODO understand why disentanglement failure sometimes happens (all 3 latent dimensions are effectively used)
# TODO implement interactive latent space navigation
# TODO the model obviously lacks the ability to deal with the symetric nature of the 3D space : neighboring states in the latent space might be
# diametrically opposed : neighboring output may share same silhouette while one is in the shadowed while the other is in plain light
# Maybe use 3D steerable CNN as model
# TODO understand how a spherical feature space can be learned, and how the latent parameter boundaries are determined
# TODO maybe train encoder and decoder separately, encoder first to force camera position. But in this case is it even relevant to do
# unsupervised learning ? enforcing a feature encoding between specific values and being able to merege and separate two features at will
# would be more appealing
# Todo code unicity :train an alternative encoder from camera position. If it can not be trained afterwards,
# try to encourage code unicity directly during training (same loss function as the alternative encoder to add to encoder loss, not to
# decoder loss)

import torch.optim as optim
from utils import cuda
from model import Position_auxiliary_encoder
import argparse

import numpy as np
import torch

from solver import Solver
from utils import str2bool

from navigate_latent_space import latent_space_navigator
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def main(args):
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    net = Solver(args)

    net.net_mode(train=False)

    # check that the model is correctly loaded
    # navigator = latent_space_navigator(net, "explore_angle")
    # navigator.navigate()
    # Training auxiliary position encoder
    # net.position_auxiliary_encoder_train()

    if args.train:
        net.auxiliary_training()
        # net.supervised_training()
    else:
        net.viz_traverse()
    if args.navigate_latent_space:
        net.net_mode(train=False)
        navigator = latent_space_navigator(net, mode="explore_latent")
        navigator.navigate()

    reinit_position_encoder = True
    if reinit_position_encoder:
        # Reinit position encoder and optimizer, for instance to change its design
        net.position_encoder = Position_auxiliary_encoder(
            2, net.z_dim)
        net.position_encoder = cuda(net.position_encoder, True)
        net.position_optim = optim.Adadelta(
            params=net.position_encoder.parameters())

        # Training auxiliary position encoder
    net.position_auxiliary_encoder_train()

    if args.navigate_latent_space:
        net.net_mode(train=False)
        navigator = latent_space_navigator(net, mode="explore_angle")
        navigator.navigate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='toy Beta-VAE')

    parser.add_argument('--train', default=True,
                        type=str2bool, help='train or traverse')
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--cuda', default=True,
                        type=str2bool, help='enable cuda')
    parser.add_argument('--max_iter', default=1e4, type=float,
                        help='maximum training iteration')
    parser.add_argument('--batch_size', default=64,
                        type=int, help='batch size')

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
    parser.add_argument('--dataset', default='CelebA',
                        type=str, help='dataset name')
    parser.add_argument('--image_size', default=64, type=int,
                        help='image size. now only (64,64) is supported')
    parser.add_argument('--num_workers', default=2,
                        type=int, help='dataloader num_workers')

    parser.add_argument('--viz_on', default=True,
                        type=str2bool, help='enable visdom visualization')
    parser.add_argument('--viz_name', default='main',
                        type=str, help='visdom env name')
    parser.add_argument('--viz_port', default=8097,
                        type=str, help='visdom port number')
    parser.add_argument('--save_output', default=True,
                        type=str2bool, help='save traverse images and gif')
    parser.add_argument('--output_dir', default='outputs',
                        type=str, help='output directory')
    parser.add_argument('--navigate_latent_space', default=True, type=str2bool,
                        help='Activate this for interactive latent space navigation after training')

    parser.add_argument('--gather_step', default=500, type=int,
                        help='numer of iterations after which data is gathered for visdom')
    parser.add_argument('--display_step', default=500, type=int,
                        help='number of iterations after which loss data is printed and visdom is updated')
    parser.add_argument('--save_step', default=500, type=int,
                        help='number of iterations after which a checkpoint is saved')

    parser.add_argument('--ckpt_dir', default='checkpoints',
                        type=str, help='checkpoint directory')
    parser.add_argument('--ckpt_name', default='last', type=str,
                        help='load previous checkpoint. insert checkpoint filename')

    parser.add_argument('--prefix', default='initial_training', type=str,
                        help='prefix directory for saving intermediate models when training iteratively')
    parser.add_argument('--base', default='initial_training', type=str,
                        help='prefix directory for previously trained model on which to base training')
    args = parser.parse_args()

    main(args)
