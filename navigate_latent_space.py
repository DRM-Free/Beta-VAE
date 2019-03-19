from tkinter import Tk, Label, Canvas, PhotoImage, E
from solver import Solver
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from utils import cuda, grid2gif, get_image

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image, ImageTk
import random


class latent_space_navigator(object):
    def __init__(self, sol):
        self.current_navigated_dim = 0
        self.navigation_step = 0.1
        self.fen1 = Tk()
        self.sol = sol
        # création de widgets Label(), Entry(), et Checkbutton() :
        Label(self.fen1, text='Controls :').grid(sticky=E)
        Label(self.fen1, text='Change exploration step : +-').grid(sticky=E)
        Label(self.fen1, text='Change explored latent dimension : up down arrrows').grid(
            sticky=E)
        Label(self.fen1, text='Explore dimension : left right arrows').grid(sticky=E)

        Label(self.fen1, text='Info:').grid(sticky=E)
        Label(self.fen1, text='Currently explored dimension : {0}'.format(
            self.current_navigated_dim)).grid(sticky=E)
        Label(self.fen1, text='Current step value :{0}'.format(
            self.navigation_step)).grid(sticky=E)

        # création d'un widget 'Canvas' contenant une image bitmap :
        self.can1 = Canvas(self.fen1, width=160, height=160, bg='white')
        sol.net_mode(train=False)

        decoder = sol.net.decoder
        encoder = sol.net.encoder

        init_img = sol.data_loader.dataset.__getitem__(0)
        init_img = Variable(
            cuda(init_img, sol.use_cuda), volatile=True).unsqueeze(0)

        self.can1.grid(row=0, column=2, rowspan=4, padx=10, pady=5)
        self.fen1.bind('<Left>', self.leftKey)
        self.fen1.bind('<Right>', self.rightKey)
        self.update_img()

    def leftKey(self, event):
        print("Left key pressed")

    def rightKey(self, event):
        print("Right key pressed")

    def update_img(self):
        self.latent_position = torch.tensor(
            [random.uniform(-2, 2)]*self.sol.z_dim).cuda()
        self.img = F.sigmoid(
            self.sol.net.decoder(self.latent_position)).data
        self.img = get_image(tensor=self.img.cpu())

        self.can1.delete()  # deletes all graphical items from can1
        # TODO fix the image here
        self.can1.create_image(
            64, 64, image=ImageTk.PhotoImage(image=self.img))

    def navigate(self):
        # démarrage :
        self.fen1.mainloop()
