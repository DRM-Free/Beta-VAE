"""solver.py"""

import warnings
warnings.filterwarnings("ignore")

import os
from tqdm import tqdm
import visdom

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import make_grid, save_image

from utils import cuda, grid2gif, get_image
from model import BetaVAE_H, BetaVAE_B, Auxiliary_network, reparametrize
from dataset import get_image_dataloader, get_pairwise_dataloader

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np


def reconstruction_loss(x, x_recon, distribution):
    batch_size = x.size(0)
    assert batch_size != 0

    if distribution == 'bernoulli':
        recon_loss = F.binary_cross_entropy_with_logits(
            x_recon, x, size_average=False).div(batch_size)
    elif distribution == 'gaussian':
        x_recon = F.sigmoid(x_recon)
        recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)
    else:
        recon_loss = None

    return recon_loss


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


class DataGather(object):
    def __init__(self):
        self.data = self.get_empty_data_dict()

    def get_empty_data_dict(self):
        return dict(iter=[],
                    recon_loss=[],
                    total_kld=[],
                    dim_wise_kld=[],
                    mean_kld=[],
                    mu=[],
                    var=[],
                    images=[],)

    def insert(self, **kwargs):
        for key in kwargs:
            self.data[key].append(kwargs[key])

    def flush(self):
        self.data = self.get_empty_data_dict()


class Solver(object):
    def __init__(self, args):
        self.args = args
        self.use_cuda = True
        self.use_cuda = args.cuda and torch.cuda.is_available()
        self.max_iter = args.max_iter
        self.global_iter = 0
        self.z_dim = args.z_dim
        self.beta = args.beta
        self.gamma = args.gamma
        self.C_max = args.C_max
        self.C_stop_iter = args.C_stop_iter
        self.objective = args.objective
        self.model = args.model
        self.lr = args.lr
        self.beta1 = args.beta1
        self.beta2 = args.beta2

        if args.dataset.lower() == 'cube_small':
            self.nc = 3
            self.decoder_dist = 'gaussian'
        elif args.dataset.lower() == 'cube_random_spherical_position':
            self.nc = 3
            self.decoder_dist = 'gaussian'
        elif args.dataset.lower() == 'two_balls':
            self.nc = 1
            self.decoder_dist = 'bernoulli'
        else:
            raise NotImplementedError

        # Create main network
        if args.model == 'H':
            VAE_net = BetaVAE_H
        elif args.model == 'B':
            VAE_net = BetaVAE_B
        else:
            raise NotImplementedError('only support model H or B')
        self.VAE_net = cuda(VAE_net(self.z_dim, self.nc), self.use_cuda)

        # Maybe add option for optionnal auxiliary network
        # Create auxiliary network
        self.Auxiliary_net = cuda(Auxiliary_network(self.z_dim), self.use_cuda)
        self.Adv_net = cuda(Auxiliary_network(self.z_dim), self.use_cuda)
        self.VAE_optim = optim.Adadelta(params=self.VAE_net.parameters())
        self.Aux_optim = optim.Adadelta(params=self.Auxiliary_net.parameters())
        self.Adv_optim = optim.Adadelta(params=self.Adv_net.parameters())
        self.viz_name = args.viz_name
        self.viz_port = args.viz_port
        self.viz_on = args.viz_on
        self.win_recon = None
        self.win_kld = None
        self.win_mu = None
        self.win_var = None
        if self.viz_on:
            self.viz = visdom.Visdom(port=self.viz_port)

        self.ckpt_dir = os.path.join(args.ckpt_dir, args.viz_name)
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir, exist_ok=True)
        self.ckpt_name = args.ckpt_name
        if self.ckpt_name is not None:
            self.load_checkpoint(self.ckpt_name)

        self.save_output = args.save_output
        self.output_dir = os.path.join(args.output_dir, args.viz_name)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        self.gather_step = args.gather_step
        self.display_step = args.display_step
        self.save_step = args.save_step

        self.dset_dir = args.dset_dir
        self.dataset = args.dataset
        self.batch_size = args.batch_size
        # Initial image data is accessed by Auxiliary dataset from VAE dataset
        self.VAE_data_loader, im_dataset = get_image_dataloader(args)
        self.Aux_data_loader = get_pairwise_dataloader(im_dataset, args)
        self.im_dataset = im_dataset
        self.gather = DataGather()

    def init_train(self):
        self.net_mode(train=True)
        self.C_max = Variable(
            cuda(torch.FloatTensor([self.C_max]), self.use_cuda))

        pbar = tqdm(total=self.max_iter)
        pbar.update(self.global_iter)
        return pbar

    def end_train(self, pbar):
        pbar.write("[Training Finished]")
        pbar.close()

    def train(self, pbar, num_steps):
        for i in range(num_steps):
            for x in self.VAE_data_loader:
                x = Variable(cuda(x, self.use_cuda))
                x_recon, mu, logvar = self.VAE_net(x)
                recon_loss = reconstruction_loss(x, x_recon, self.decoder_dist)
                total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)

                if self.objective == 'H':
                    beta_vae_loss = recon_loss + self.beta*total_kld
                elif self.objective == 'B':
                    C = torch.clamp(self.C_max/self.C_stop_iter *
                                    self.global_iter, 0, self.C_max.data[0])
                    beta_vae_loss = recon_loss + self.gamma*(total_kld-C).abs()

                self.VAE_optim.zero_grad()
                beta_vae_loss.backward()
                self.VAE_optim.step()

                if self.viz_on and self.global_iter % self.gather_step == 0:
                    self.gather.insert(iter=self.global_iter,
                                       mu=mu.mean(0).data, var=logvar.exp().mean(
                                           0).data,
                                       recon_loss=recon_loss.data, total_kld=total_kld.data,
                                       dim_wise_kld=dim_wise_kld.data, mean_kld=mean_kld.data)
            self.global_iter += 1
            pbar.update(1)

    def auxiliary_training(self):
        # TODO make sure initial disambiguation is done only if the network was not previously trained
        # self.initial_disambiguation()
        pbar = self.init_train()
        nb_phases = int(10)
        max_iter = self.max_iter
        num_steps = int(np.ceil(max_iter/nb_phases))
        for training_phase in range(nb_phases):

            # Deactivate gradients for decoder part for auxiliary training
            self.deactivate_grad(self.VAE_net.decoder)
            self.encoder_auxiliary_training()
            self.encoder_adversarial_training()
            # Reactivate gradients for decoder part
            self.activate_grad(self.VAE_net.decoder)

            self.train(pbar, num_steps)
            if training_phase == nb_phases:
                save_name = "last"
            else:
                save_name = "phase_{}".format(training_phase)
                # TODO check that auxiliary network is well saved and loaded
            self.save_checkpoint(save_name)
            self.viz_traverse()
            # Pick new images for auxiliary and adversarial training (this is expensive due to all the pairwise ambiguity/similarity computations)
            self.Aux_data_loader.dataset.sample_images()
        self.end_train(pbar)

    def initial_disambiguation(self):
        print("initial disambiguation...")
        for i in range(3):
            self.encoder_auxiliary_training()
            self.encoder_adversarial_training()
            self.Aux_data_loader.dataset.sample_images()
        print("done")

    # Adversarial or Cooperative training of VAE encoder and auxiliary network
    # TODO verify proper shuffling of data
    # TODO verify autograd : is the gradient properly computed ?
    def encoder_auxiliary_training(self):
        # Pairwise ambiguities are expensive to compute, so we use it several times each epoch
        # TODO add range here to reuse supervised data
        for _, data in enumerate(self.Aux_data_loader, 0):
            imgs, ambiguity, similarity = data
            # for ambiguity, imgs in self.Aux_data_loader:
            # TODO check images are passed properly by dataloader
            img0 = imgs[0].cuda()
            img1 = imgs[1].cuda()
            # why is the code of size 6 instead of 3 ? Why is the reparametrize function ?
            code1 = self.VAE_net._encode(img0)
            mu = code1[:, :self.z_dim]
            logvar = code1[:, self.z_dim:]
            code1 = reparametrize(mu, logvar)
            code2 = self.VAE_net._encode(img1)
            mu = code2[:, :self.z_dim]
            logvar = code2[:, self.z_dim:]
            code2 = reparametrize(mu, logvar)

            # First dimension (0) is batch size so we cat in dim 1
            AUX_net_input = torch.cat((code1, code2), 1)
            Aux_output = self.Auxiliary_net.forward(AUX_net_input)

            similarity = cuda(similarity.float(), self.use_cuda)
            similarity.requires_grad = True
            Aux_loss = F.mse_loss(
                Aux_output, similarity, size_average=False).div(self.batch_size)
            # Aux_loss = F.cross_entropy(Aux_output, similarity)
            self.Aux_optim.zero_grad()
            self.VAE_optim.zero_grad()
            # TODO make sure this loss is good for the selected task (adversarial, auxiliary) and is applied to both trainings
            # Auxiliary loss is shared between encoder and auxiliary net
            Aux_loss.backward()
            # TODO check that these steps update the required weights
            self.Aux_optim.step()
            self.VAE_optim.step()
        # Refresh pairwise dataset from image dataset
            self.Aux_data_loader = get_pairwise_dataloader(
                self.im_dataset, self.args)

    def encoder_adversarial_training(self):
        for _, data in enumerate(self.Aux_data_loader, 0):
            imgs, ambiguity, similarity = data
            # for ambiguity, imgs in self.Aux_data_loader:
            # TODO check images are passed properly by dataloader
            img0 = imgs[0].cuda()
            img1 = imgs[1].cuda()
            # why is the code of size 6 instead of 3 ? Why is the reparametrize function ?
            code1 = self.VAE_net._encode(img0)
            mu = code1[:, :self.z_dim]
            logvar = code1[:, self.z_dim:]
            code1 = reparametrize(mu, logvar)
            code2 = self.VAE_net._encode(img1)
            mu = code2[:, :self.z_dim]
            logvar = code2[:, self.z_dim:]
            code2 = reparametrize(mu, logvar)

            Adv_net_input = torch.cat((code1, code2), 1)
            Adv_output = self.Adv_net.forward(Adv_net_input)
            ambiguity = cuda(ambiguity.float(), self.use_cuda)
            ambiguity.requires_grad = True
            ADV_loss = F.mse_loss(Adv_output, ambiguity,
                                  size_average=False).div(self.batch_size)

            # self.Adv_optim.zero_grad()
            # self.VAE_optim.zero_grad()
            # ADV_loss.backward()
            # self.Adv_optim.step()
            # self.VAE_optim.step()
            ######################### Keep these lines#########################
            # ADV network aims at finding ambiguous pairs of images based on their code
            # VAE encoder will try to prevent this by resorting to unambiguous encoding
            VAE_loss = -ADV_loss
            # TODO check that these steps update the required weights
            # TODO make sure this loss is good for the selected task (adversarial / auxiliary)
            self.deactivate_grad(self.VAE_net.encoder)
            self.Adv_optim.zero_grad()
            ADV_loss.backward(retain_graph=True)
            self.Adv_optim.step()
            self.activate_grad(self.VAE_net.encoder)

            self.deactivate_grad(self.Adv_net.net)
            self.VAE_optim.zero_grad()
            VAE_loss.backward()
            self.VAE_optim.step()
            self.activate_grad(self.Adv_net.net)
            ######################### Keep these lines#########################

        # Refresh pairwise dataset from image dataset
            self.Aux_data_loader = get_pairwise_dataloader(
                self.im_dataset, self.args)

    def deactivate_grad(self, net):
        for param in net.parameters():
            param.requires_grad = False

    def activate_grad(self, net):
        for param in net.parameters():
            param.requires_grad = True

    def viz_reconstruction(self):
        self.net_mode(train=False)
        x = self.gather.data['images'][0][:100]
        x = make_grid(x, normalize=True)
        x_recon = self.gather.data['images'][1][:100]
        x_recon = make_grid(x_recon, normalize=True)
        images = torch.stack([x, x_recon], dim=0).cpu()
        self.viz.images(images, env=self.viz_name+'_reconstruction',
                        opts=dict(title=str(self.global_iter)), nrow=10)
        self.net_mode(train=True)

    def viz_traverse(self, limit=3, inter=0.1, loc=-1):
        self.net_mode(train=False)
        import random

        decoder = self.VAE_net.decoder
        encoder = self.VAE_net.encoder
        interpolation = torch.arange(-limit, limit+0.1, inter)

        n_dsets = len(self.VAE_data_loader.dataset)
        rand_idx = random.randint(1, n_dsets-1)

        random_img = self.VAE_data_loader.dataset.__getitem__(rand_idx)
        random_img = Variable(
            cuda(random_img, self.use_cuda), volatile=True).unsqueeze(0)
        random_img_z = encoder(random_img)[:, :self.z_dim]

        random_z = Variable(cuda(torch.rand(1, self.z_dim),
                                 self.use_cuda), volatile=True)
        fixed_idx = 0
        fixed_img = self.VAE_data_loader.dataset.__getitem__(fixed_idx)
        fixed_img = Variable(
            cuda(fixed_img, self.use_cuda), volatile=True).unsqueeze(0)
        fixed_img_z = encoder(fixed_img)[:, :self.z_dim]

        Z = {'fixed_img': fixed_img_z,
             'random_img': random_img_z, 'random_z': random_z}

        gifs = []
        for key in Z.keys():
            z_ori = Z[key]
            samples = []
            for row in range(self.z_dim):
                if loc != -1 and row != loc:
                    continue
                z = z_ori.clone()
                for val in interpolation:
                    loc
                    z[:, row] = val
                    sample = F.sigmoid(decoder(z)).data
                    samples.append(sample)
                    gifs.append(sample)
            samples = torch.cat(samples, dim=0).cpu()
            title = '{}_latent_traversal(iter:{})'.format(
                key, self.global_iter)

        if self.save_output:
            output_dir = os.path.join(self.output_dir, str(self.global_iter))
            os.makedirs(output_dir, exist_ok=True)
            gifs = torch.cat(gifs)
            gifs = gifs.view(len(Z), self.z_dim, len(
                interpolation), self.nc, 64, 64).transpose(1, 2)
            for i, key in enumerate(Z.keys()):
                for j, val in enumerate(interpolation):
                    save_image(tensor=gifs[i][j].cpu(),
                               filename=os.path.join(
                                   output_dir, '{}_{}.jpg'.format(key, j)),
                               nrow=self.z_dim, pad_value=1)

                grid2gif(os.path.join(output_dir, key+'*.jpg'),
                         os.path.join(output_dir, key+'.gif'), delay=10)

        self.net_mode(train=True)

    def net_mode(self, train):
        if not isinstance(train, bool):
            raise('Only bool type is supported. True or False')

        if train:
            self.VAE_net.train()
            self.Auxiliary_net.train()
            self.Adv_net.train()
        else:
            self.VAE_net.eval()
            self.Auxiliary_net.eval()
            self.Adv_net.eval()

    def save_checkpoint(self, filename, silent=True):
        model_states = {'VAE_net': self.VAE_net.state_dict(
        ), 'AUX_net': self.Auxiliary_net.state_dict()}
        optim_states = {'VAE_optim': self.VAE_optim.state_dict(
        ), 'Aux_optim': self.Aux_optim.state_dict(), }
        win_states = {'recon': self.win_recon,
                      'kld': self.win_kld,
                      'mu': self.win_mu,
                      'var': self.win_var, }
        states = {'iter': self.global_iter,
                  'win_states': win_states,
                  'model_states': model_states,
                  'optim_states': optim_states}

        file_path = os.path.join(self.ckpt_dir, filename)
        with open(file_path, mode='wb+') as f:
            torch.save(states, f)
        if not silent:
            print("=> saved checkpoint '{}' (iter {})".format(
                file_path, self.global_iter))

    def load_checkpoint(self, filename):
        file_path = os.path.join(self.ckpt_dir, filename)
        if os.path.isfile(file_path):
            checkpoint = torch.load(file_path)
            self.global_iter = checkpoint['iter']
            self.win_recon = checkpoint['win_states']['recon']
            self.win_kld = checkpoint['win_states']['kld']
            self.win_var = checkpoint['win_states']['var']
            self.win_mu = checkpoint['win_states']['mu']
            self.VAE_net.load_state_dict(checkpoint['model_states']['VAE_net'])
            self.VAE_optim.load_state_dict(
                checkpoint['optim_states']['VAE_optim'])
            self.Auxiliary_net.load_state_dict(
                checkpoint['model_states']['AUX_net'])
            self.Aux_optim.load_state_dict(
                checkpoint['optim_states']['Aux_optim'])
            print("=> loaded checkpoint '{} (iter {})'".format(
                file_path, self.global_iter))
        else:
            print("=> no checkpoint found at '{}'".format(file_path))
