from tkinter import Tk, Label, Canvas, PhotoImage, E, LEFT, Frame
from solver import Solver
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from utils import cuda, grid2gif, get_image

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image, ImageTk
from random import randint, uniform
import numpy as np

# TODO add support for real alongside latent position. Recommended usage of a position manager


class latent_space_navigator(object):
    def __init__(self, sol, mode):
        self.mode = mode
        self.current_navigated_dim = 0
        self.navigation_step = 0.5
        self.displayed_image_size = [500, 500]
        self.fen1 = Tk()
        self.sol = sol
        self.sol.position_encoder.train(False)
        self.sol.deactivate_grad(self.sol.position_encoder.net)
        if self.mode == "explore_latent":
            self.max_dim = self.sol.z_dim-1
        elif self.mode == "explore_angle":
            self.max_dim = 1
        # création de widgets Label()
        Labels_frame = Frame(self.fen1)
        Labels_frame.grid(sticky=E, row=0, column=0)
        Label(Labels_frame, text='Controls :').pack()
        Label(Labels_frame, text='Reinit latent position : Keypad zero').pack()
        Label(Labels_frame, text='Change exploration step : +-').pack()
        Label(Labels_frame,
              text='Change explored latent dimension : up down arrrows').pack()
        Label(Labels_frame, text='Explore dimension : left right arrows').pack()
        if self.mode == "explore_latent":
            text = "exploring latent space"
        elif self.mode == "explore_angle":
            text = "exploring angle space"
        Label(Labels_frame, text=text).pack()

        Label(Labels_frame, text='Info:').pack()
        Label(Labels_frame, text='Latent space dimension : {0}'.format(
            self.sol.z_dim)).pack()
        # Next labels need updating so give them a name and keep them in self
        self.info_dim = Label(Labels_frame, text='Currently explored dimension : {0}'.format(
            self.current_navigated_dim))
        self.info_dim.pack()
        self.info_step = Label(Labels_frame, text='Current step value :{0}'.format(
            self.navigation_step))
        self.info_step.pack()
        self.info_latent_pos = Label(
            Labels_frame, text='Current latent position :')
        self.info_latent_pos.pack()
        self.info_angle_pos = Label(
            Labels_frame, text='Current angle position :')
        self.info_angle_pos.pack()

        if self.mode == "explore_latent":
            self.fen1.bind('<Left>', self.navigate_latent_left)
            self.fen1.bind('<Right>', self.navigate_latent_right)
            self.fen1.bind('<Key-KP_0>', self.reinit_image)
        elif self.mode == "explore_angle":
            self.fen1.bind('<Left>', self.navigate_angle_left)
            self.fen1.bind('<Right>', self.navigate_angle_right)
            self.fen1.bind('<Key-KP_0>', self.reinit_angle)
        self.fen1.bind('<Up>', self.eplored_dimension_increment)
        self.fen1.bind('<Down>', self.explored_dimension_decrement)
        self.fen1.bind('<Key-KP_Add>', self.increment_exploration_step)
        self.fen1.bind('<Key-KP_Subtract>', self.decrement_exploration_step)
        self.fen1.bind('<Key-Escape>', self.close_window)

        # self.init_latent_position() #for latent space navigation only
        self.init_angle_position()
        self.init_img()
        self.update_img()
        self.update_info_latent_pos()

    def navigate_latent_left(self, event):
        self.latent_position[self.current_navigated_dim] = self.latent_position[self.current_navigated_dim] - self.navigation_step
        self.update_img()
        self.update_info_latent_pos()

    def navigate_latent_right(self, event):
        self.latent_position[self.current_navigated_dim] = self.latent_position[self.current_navigated_dim] + self.navigation_step
        self.update_img()
        self.update_info_latent_pos()

    def navigate_angle_left(self, event):
        self.angle_position[self.current_navigated_dim] = self.angle_position[self.current_navigated_dim] - self.navigation_step
        self.guess_latent_code()
        self.update_img()
        self.update_info_angle_pos()
        self.update_info_latent_pos()

    def navigate_angle_right(self, event):
        self.angle_position[self.current_navigated_dim] = self.angle_position[self.current_navigated_dim] + self.navigation_step
        self.guess_latent_code()
        self.update_img()
        self.update_info_angle_pos()
        self.update_info_latent_pos()

    def eplored_dimension_increment(self, event):
        if self.mode == "explore_latent":
            self.current_navigated_dim = min(
                self.current_navigated_dim + 1, self.max_dim)
            self.update_info_dim()
        elif self.mode == "explore_angle":
            self.current_navigated_dim = min(
                self.current_navigated_dim + 1, 1)
            self.update_info_dim()
        print("Explored dimension updated")

    def explored_dimension_decrement(self, event):
        self.current_navigated_dim = max(
            self.current_navigated_dim - 1, 0)
        self.update_info_dim()
        print("Explored dimension updated")

    def increment_exploration_step(self, event):
        self.navigation_step = self.navigation_step + 0.1
        self.update_info_step()
        print("Exploration step increased")

    def decrement_exploration_step(self, event):
        self.navigation_step = self.navigation_step - 0.1
        self.update_info_step()
        print("Exploration step decreased")

    def update_info_step(self):
        self.info_step.configure(
            text='Current step value :{0:.2f}'.format(self.navigation_step))

    def update_info_dim(self):
        self.info_dim.configure(
            text='Current navigated dimension :{}'.format(self.current_navigated_dim))

    def update_info_latent_pos(self):
        self.info_latent_pos.configure(text='Current latent position :{}'.format(
            np.around(self.latent_position, 2)))

    def update_info_angle_pos(self):
        self.info_angle_pos.configure(text='Current angle position :{}'.format(
            np.around(self.angle_position, 2)))

    def reinit_image(self, event):
        self.init_latent_position()
        self.update_img()
        self.update_info_latent_pos()
        print("Latent position re-initialized")

    def reinit_angle(self, event):
        self.init_angle_position()
        self.update_img()
        self.update_info_angle_pos()
        self.update_info_latent_pos()
        print("Latent position re-initialized")

    def init_latent_position(self):
        # print(self.sol.data_loader.dataset.__len__())
        # get one random image from dataset
        ind = randint(1, self.sol.VAE_data_loader.dataset.__len__())
        tensor_image, pos = self.sol.VAE_data_loader.dataset.__getitem__(ind)

        # encode this image for initial latent position
        tensor_image = Variable(
            cuda(tensor_image, self.sol.use_cuda), volatile=True).unsqueeze(0)
        self.latent_position = self.encode_image(tensor_image)

        # send initial image to CPU
        self.img_arr = get_image(tensor=tensor_image.cpu())

    def init_angle_position(self):
        self.angle_position = [0., 0.]
        self.guess_latent_code()
        self.img_arr = F.sigmoid(
            self.sol.VAE_net.decoder(cuda(self.latent_position, True)).cpu().data)
        self.img_arr = get_image(tensor=self.img_arr.cpu())

    def guess_latent_code(self):
        position_tensor = cuda(torch.tensor(self.angle_position), True)
        self.latent_position = self.sol.position_encoder.forward(
            position_tensor)
        self.latent_position = self.latent_position.cpu()

    def encode_image(self, image):
        latent_position = self.sol.VAE_net.encoder(image)[:, :self.sol.z_dim]
        latent_position = latent_position.data.cpu().numpy()[0]
        return latent_position

    def update_img(self):
        position_tensor = torch.tensor(self.latent_position).cuda()
        self.img_arr = F.sigmoid(
            self.sol.VAE_net.decoder(position_tensor).data)
        self.img_arr = get_image(tensor=self.img_arr.cpu())
        self.scale_img()
        img = ImageTk.PhotoImage(image=self.img_arr)
        self.label.configure(image=img)
        self.label.image = img  # keeping a reference to prevent garbage collection of img
        # self.label.update()
        # self.label.update_idletasks()
        self.fen1.update_idletasks()

    def init_img(self):
        self.scale_img()
        img = ImageTk.PhotoImage(image=self.img_arr)
        # self.label = Label(self.can1, image=img)
        self.label = Label(self.fen1, image=img)
        self.label.image = img  # keep a reference!
        # put the label inside parent window
        self.label.grid(sticky=E, row=0, column=1)

    def scale_img(self):
        self.img_arr = self.img_arr.resize(self.displayed_image_size)

    def close_window(self, event):
        self.fen1.destroy()

    def navigate(self):
        # démarrage :
        self.fen1.mainloop()
