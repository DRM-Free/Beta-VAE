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
from model import BetaVAE_H, BetaVAE_B, Auxiliary_network, Position_auxiliary_encoder, reparametrize
from dataset import get_image_dataloader, get_pairwise_dataloader

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np

from math import atan2, acos, sqrt


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
        self.prefix = args.prefix
        self.base=args.base

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
        # TODO set position size automatically
        pos_size = 2
        self.position_encoder = Position_auxiliary_encoder(pos_size, self.z_dim)
        self.position_encoder = cuda(self.position_encoder, True)
        self.position_optim = optim.Adadelta(params=self.position_encoder.parameters(),lr=0.1)

        self.VAE_optim = optim.Adadelta(params=self.VAE_net.parameters())
        self.VAE_aux_optim = optim.Adadelta(params=self.VAE_net.encoder.parameters())
        self.VAE_adv_optim = optim.Adadelta(
            params=self.VAE_net.encoder.parameters())

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
        self.similarity_data_loader = get_pairwise_dataloader(im_dataset, args, type="similarity")
        self.ambiguity_data_loader = get_pairwise_dataloader(im_dataset, args, type="ambiguity")

        self.im_dataset = im_dataset
        self.gather = DataGather()

    def init_train(self, nb_phases):
        self.net_mode(train=True)
        self.C_max = Variable(
            cuda(torch.FloatTensor([self.C_max]), self.use_cuda))

        pbar = tqdm(total=nb_phases)
        return pbar

    def end_train(self, pbar):
        pbar.write("[Training Finished]")
        pbar.close()
        self.net_mode(train=False)

    def train(self, num_steps):
        for i in range(num_steps):
            for x, pos in self.VAE_data_loader:
                x = Variable(cuda(x, self.use_cuda))
                x_recon, mu, logvar = self.VAE_net(x)
                recon_loss = reconstruction_loss(x, x_recon, self.decoder_dist)
                total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)

                if self.objective == 'H':
                    beta_vae_loss = recon_loss + self.beta*total_kld
                elif self.objective == 'B':
                    C = torch.clamp(self.C_max/self.C_stop_iter *
                                    self.global_iter, 0, self.C_max.data[0])
                    beta_vae_loss = recon_loss + self.gamma * (total_kld - C).abs()
                self.VAE_step(beta_vae_loss)

    def train_decoder_alone(self, num_steps):
        self.deactivate_grad(self.VAE_net.encoder)
        self.train(num_steps)
        self.activate_grad(self.VAE_net.encoder)

    def train_encoder_alone(self, num_steps):
        self.deactivate_grad(self.VAE_net.decoder)
        self.train(num_steps)
        self.activate_grad(self.VAE_net.decoder)

    def auxiliary_training(self):
        # TODO make sure initial disambiguation is done only if the network was not previously trained
        # self.initial_disambiguation()
        nb_phases = int(10)
        pbar = self.init_train(nb_phases)
        max_iter = self.max_iter
        num_steps = int(np.ceil(max_iter/nb_phases))
        for training_phase in range(nb_phases):

            # Deactivate gradients for decoder part for auxiliary training
            self.deactivate_grad(self.VAE_net.decoder)
            self.encoder_auxiliary_training(num_steps)
            self.encoder_adversarial_training(num_steps)
            # Reactivate gradients for decoder part
            self.activate_grad(self.VAE_net.decoder)
            # Train decoder alone in order not to lose disambiguation achieved with encoder
            self.train_decoder_alone(num_steps)
            self.train(num_steps)
            self.global_iter+=1

            if training_phase == nb_phases-1:
                save_name = "last"
            else:
                save_name = "phase_"+str(training_phase)
                # TODO check that auxiliary network is well saved and loaded
            self.save_checkpoint(save_name)
            self.viz_traverse()
            pbar.update(1)
        self.end_train(pbar)

    
    def supervised_training(self):
        self.deactivate_grad(self.VAE_net.encoder)
        nb_phases = int(10)
        pbar = self.init_train(nb_phases=nb_phases)
        for training_phase in range(nb_phases):
            for img, pos in self.VAE_data_loader:
                img = Variable(cuda(img, self.use_cuda))

                x_batch = pos["x"].numpy()
                y_batch = pos["y"].numpy()
                z_batch = pos["z"].numpy()
                angle_pos = np.zeros((np.size(x_batch), self.z_dim))
                # pos.shape[0] is batch size
                for batch_elt in range(np.size(x_batch)):
                    x = x_batch[batch_elt]
                    y = y_batch[batch_elt]
                    z = z_batch[batch_elt]
                    theta = acos(z/sqrt(x**2+y**2+z**2))
                    phi = atan2(y, x)
                    angle_pos[batch_elt] = [theta, phi]
                angle_pos = cuda(torch.tensor(angle_pos).float(), True)
                # pos = Variable(cuda(pos, self.use_cuda)) #if we want to use cartesian pos instead of angle pos
                img_recon = self.VAE_net._decode(angle_pos)
                VAE_loss = F.mse_loss(img_recon,img)
                self.VAE_step(VAE_loss)
                self.global_iter += 1
            pbar.update(1)
            if training_phase == nb_phases-1:
                save_name = "last"
            else:
                save_name = "phase_"+str(training_phase)
            self.save_checkpoint(save_name)
            self.viz_traverse()
        self.end_train(pbar)


    def position_auxiliary_encoder_train(self):

        self.position_encoder.train()
        self.net_mode(train=False)
        nb_epoch = 100
        pbar = tqdm(total=nb_epoch)
        for i in range(nb_epoch):
            for img, pos in self.VAE_data_loader:
                img = Variable(cuda(img, self.use_cuda))

                x_batch = pos["x"].numpy()
                y_batch = pos["y"].numpy()
                z_batch = pos["z"].numpy()
                angle_pos = np.zeros((np.size(x_batch), self.z_dim))
                # pos.shape[0] is batch size
                for batch_elt in range(np.size(x_batch)):
                    x = x_batch[batch_elt]
                    y = y_batch[batch_elt]
                    z = z_batch[batch_elt]
                    theta = acos(z/sqrt(x**2+y**2+z**2))
                    phi = atan2(y, x)
                    angle_pos[batch_elt] = [theta, phi]
                angle_pos = cuda(torch.tensor(angle_pos).float(), True)
                angle_pos.requires_grad=True
                with torch.no_grad():
                    img = cuda(torch.tensor(img), uses_cuda=True)
                    code = self.VAE_net._encode(img)  # shape batch size * 6
                    mu = code[:, :self.z_dim]
                    logvar = code[:, self.z_dim:]
                    code = reparametrize(mu, logvar)
                # code = cuda(torch.tensor(code), uses_cuda=True)
                # This would not work if code was sent to GPU, as encoder was not sent to gpu
                # Use angle positions for training !
                guess_code = self.position_encoder.forward(angle_pos)
                # guess_code = position_encoder.forward(pos)
                position_loss = F.mse_loss(
                    guess_code, code).div(self.batch_size)
                self.position_optim.zero_grad()
                position_loss.backward()
                self.position_optim.step()
                loss_str = str(position_loss.cpu().detach().numpy())
                pbar.set_description("loss: " + loss_str)

            pbar.update(1)
        pbar.close()
        self.save_checkpoint(filename="last")

        # # save auxiliary position network
        # filename = os.path.join(self.ckpt_dir, "position_auxiliary_encoder")
        # pos_states = {'position_net': position_encoder.state_dict()}
        # with open(filename, mode='wb+') as f:
        #     torch.save(pos_states, f)

    def encode_pairwise_imgs(self, data):
            im1 = data["im1"]
            im2 = data["im2"]
            # TODO check images are passed properly by dataloader
            im1 = cuda(im1, True)
            im2 = cuda(im2, True)
            # why is the code of size 6 instead of 3 ? Why is the reparametrize function ?
            code1 = self.VAE_net._encode(im1)
            mu = code1[:, :self.z_dim]
            logvar = code1[:, self.z_dim:]
            code1 = reparametrize(mu, logvar)

            code2 = self.VAE_net._encode(im2)
            mu = code2[:, :self.z_dim]
            logvar = code2[:, self.z_dim:]
            code2 = reparametrize(mu, logvar)
            return code1, code2

    def aux_loss(self, code1, code2, similarity):
        # First dimension (0) is batch size so we cat in dim 1
        AUX_net_input = torch.cat((code1, code2), 1)
        Aux_output = self.Auxiliary_net.forward(AUX_net_input)

        similarity = cuda(similarity.float(), self.use_cuda)
        Aux_loss = F.mse_loss(Aux_output, similarity, size_average=False).div(self.batch_size)
        return Aux_loss

    def adv_loss(self, code1, code2, ambiguity):
                    # First dimension (0) is batch size so we cat in dim 1
        Adv_net_input = torch.cat((code1, code2), 1)
        Adv_output = self.Adv_net.forward(Adv_net_input)

        ambiguity = cuda(ambiguity.float(), self.use_cuda)
        Adv_loss = F.mse_loss(Adv_output, ambiguity,
                              size_average=False).div(self.batch_size)
        return Adv_loss

    def VAE_step(self, VAE_loss):
        self.VAE_optim.zero_grad()
        VAE_loss.backward()
        self.VAE_optim.step()

    def VAE_aux_step(self, VAE_loss):
        self.VAE_optim.zero_grad()
        VAE_loss.backward()
        self.VAE_aux_optim.step()
    
    def VAE_adv_step(self, VAE_loss):
        self.VAE_optim.zero_grad()
        VAE_loss.backward()
        self.VAE_adv_optim.step()

    def aux_step(self, Aux_loss):
        self.Aux_optim.zero_grad()
        Aux_loss.backward()
        self.Aux_optim.step()


    def adv_step(self, Adv_loss):
        self.Adv_optim.zero_grad()
        Adv_loss.backward()
        self.Adv_optim.step()

    # Adversarial or Cooperative training of VAE encoder and auxiliary network
    # TODO verify proper shuffling of data
    # TODO verify autograd : is the gent properly computed ?

    def encoder_auxiliary_training(self, num_steps):
        # Pairwise ambiguities are expensive to compute, so we might use ithemt several times each epoch
        # Train auxiliary network then encoder

        for i in range(num_steps):
            for _, data in enumerate(self.similarity_data_loader, 0):
                similarity = data["similarity"]
                # im1, im2, ambiguity, similarity = data
                code1, code2 = self.encode_pairwise_imgs(data)
                Aux_loss = self.aux_loss(
                    code1, code2, similarity)/self.similarity_data_loader.batch_size
                self.aux_step(Aux_loss)
        # Refresh pairwise dataset from image dataset
            self.similarity_data_loader.dataset.sample_images()

        for i in range(num_steps):
            for _, data in enumerate(self.similarity_data_loader, 0):
                similarity = data["similarity"]
                # im1, im2, ambiguity, similarity = data
                code1, code2 = self.encode_pairwise_imgs(data)
                Aux_loss = self.aux_loss(code1, code2, similarity)/self.similarity_data_loader.batch_size
                self.VAE_aux_step(Aux_loss)

    def encoder_adversarial_training(self, num_steps):
        for i in range(num_steps):
            for _, data in enumerate(self.ambiguity_data_loader, 0):
                ambiguity = data["ambiguity"]
                # im1, im2, ambiguity, similarity = data
                code1, code2 = self.encode_pairwise_imgs(data)
                Adv_loss = self.adv_loss(
                    code1, code2, ambiguity)/self.ambiguity_data_loader.batch_size
                self.adv_step(Adv_loss)
            # Refresh pairwise dataset from image dataset
            self.ambiguity_data_loader.dataset.sample_images()

        for i in range(num_steps):
            for _, data in enumerate(self.ambiguity_data_loader, 0):
                ambiguity = data["ambiguity"]
                # im1, im2, ambiguity, similarity = data
                code1, code2 = self.encode_pairwise_imgs(data)
                Adv_loss = self.adv_loss(
                    code1, code2, ambiguity)/self.ambiguity_data_loader.batch_size
                # self.VAE_step(Adv_loss * -1)
                self.VAE_adv_step(Adv_loss) #for auxiliary training instead of adversarial training


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

    def viz_traverse(self,limit=3, inter=0.1, loc=-1):
        self.net_mode(train=False)
        import random

        decoder = self.VAE_net.decoder
        encoder = self.VAE_net.encoder
        interpolation = torch.arange(-limit, limit+0.1, inter)

        n_dsets = len(self.VAE_data_loader.dataset)
        rand_idx = random.randint(1, n_dsets-1)

        random_img,pos = self.VAE_data_loader.dataset.__getitem__(rand_idx)
        random_img = Variable(
            cuda(random_img, self.use_cuda), volatile=True).unsqueeze(0)
        random_img_z = encoder(random_img)[:, :self.z_dim]

        random_z = Variable(cuda(torch.rand(1, self.z_dim),
                                 self.use_cuda), volatile=True)
        fixed_idx = 0
        fixed_img,pos = self.VAE_data_loader.dataset.__getitem__(fixed_idx)
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
            output_dir = os.path.join(self.output_dir, self.prefix,str(self.global_iter))
            os.makedirs(output_dir, exist_ok=True)

            gifs = torch.cat(gifs)
            gifs = gifs.view(len(Z), self.z_dim, len(
                interpolation), self.nc, 64, 64).transpose(1, 2)
            for i, key in enumerate(Z.keys()):
                for j, val in enumerate(interpolation):
                    save_image(tensor=gifs[i][j].cpu(),
                               filename=os.path.join(
                                   output_dir,'{}_{}.jpg'.format(key, j)),
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
        ), 'AUX_net': self.Auxiliary_net.state_dict(), 'adv_net':self.Adv_net.state_dict(),'pos_net':self.position_encoder.state_dict()}
        optim_states = {'VAE_optim': self.VAE_optim.state_dict(),  'Aux_optim': self.Aux_optim.state_dict(
        ), 'Adv_optim': self.Adv_optim.state_dict(), 'vae_adv_optim': self.VAE_adv_optim.state_dict(), 'vae_aux_optim': self.VAE_aux_optim.state_dict(), 'pos_optim': self.position_optim.state_dict()}
        win_states = {'recon': self.win_recon,
                      'kld': self.win_kld,
                      'mu': self.win_mu,
                      'var': self.win_var, }
        states = {'iter': self.global_iter,
                  'win_states': win_states,
                  'model_states': model_states,
                  'optim_states': optim_states}
        if not os.path.isdir(os.path.join(self.ckpt_dir, self.prefix)):
            os.makedirs(os.path.join(self.ckpt_dir, self.prefix))

        file_path = os.path.join(self.ckpt_dir, self.prefix,filename)
        with open(file_path, mode='wb+') as f:
            torch.save(states, f)
        if not silent:
            print("=> saved checkpoint '{}' (iter {})".format(
                file_path, self.global_iter))

    def load_checkpoint(self, filename):
        file_path = os.path.join(self.ckpt_dir, self.base,filename)
        if os.path.isfile(file_path):
            checkpoint = torch.load(file_path)
            self.global_iter = checkpoint['iter']
            self.win_recon = checkpoint['win_states']['recon']
            self.win_kld = checkpoint['win_states']['kld']
            self.win_var = checkpoint['win_states']['var']
            self.win_mu = checkpoint['win_states']['mu']
            self.VAE_net.load_state_dict(checkpoint['model_states']['VAE_net'])
            self.Auxiliary_net.load_state_dict(
                checkpoint['model_states']['AUX_net'])
            self.Adv_net.load_state_dict(
                checkpoint['model_states']['adv_net'])
            self.VAE_optim.load_state_dict(
                checkpoint['optim_states']['VAE_optim'])
            self.Aux_optim.load_state_dict(
                checkpoint['optim_states']['Aux_optim'])
            self.Adv_optim.load_state_dict(
                checkpoint['optim_states']['Adv_optim'])
            self.VAE_adv_optim.load_state_dict(
                checkpoint['optim_states']['vae_adv_optim'])
            self.VAE_aux_optim.load_state_dict(
                checkpoint['optim_states']['vae_aux_optim'])

            print("=> loaded checkpoint '{} (iter {})'".format(
                file_path, self.global_iter))
            self.position_encoder.load_state_dict(
                checkpoint['model_states']["pos_net"])
            self.position_optim.load_state_dict(
                checkpoint['optim_states']['pos_optim'])
        else:
            print("=> no checkpoint found at '{}'".format(file_path))
