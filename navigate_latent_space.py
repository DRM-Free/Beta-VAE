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
import numpy as np


class latent_space_navigator(object):
    def __init__(self, sol):
        self.current_navigated_dim = 0
        self.navigation_step = 0.1
        self.displayed_image_size = [200, 200]
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
        self.can1 = Canvas(self.fen1, width=200, height=200, bg='white')
        sol.net_mode(train=False)

        self.can1.grid(row=0, column=2, rowspan=4, padx=10, pady=5)
        self.fen1.bind('<Left>', self.leftKey)
        self.fen1.bind('<Right>', self.rightKey)
        self.fen1.bind('<Up>', self.upKey)
        self.fen1.bind('<Down>', self.downKey)
        self.fen1.bind('<Key-KP_Add>', self.plusKey)
        self.fen1.bind('<Key-KP_Subtract>', self.minusKey)
        self.fen1.bind('<Key-Escape>', self.close_window)

        self.init_img()

    def leftKey(self, event):
        print("Left key pressed")

    def rightKey(self, event):
        print("Right key pressed")

    def upKey(self, event):
        print("Up key pressed")

    def downKey(self, event):
        print("Down key pressed")

    def plusKey(self, event):
        print("Plus key pressed")

    def minusKey(self, event):
        print("Minus key pressed")
        self.update_img()

    def update_img(self):
        self.latent_position = torch.tensor(
            [random.uniform(-2, 2)]*self.sol.z_dim).cuda()
        self.img = F.sigmoid(
            self.sol.net.decoder(self.latent_position)).data
        self.img = get_image(tensor=self.img.cpu())
        self.img = PhotoImage(image=self.img)
        # self.img.image = self.img
        self.scale_img()
        # self.can1.delete()  # deletes all graphical items from can1
        # TODO fix the image here
        # self.img = Image.fromarray(np.uint8(self.img))

        # self.can1.create_image(64, 64, image=self.img)
        # self.can1.create_image(64, 64, image=self.img)
        self.can1.update_idletasks

    def init_img(self):
        self.img = self.sol.data_loader.dataset.__getitem__(0)
        self.img = Variable(
            cuda(self.img, self.sol.use_cuda), volatile=True).unsqueeze(0)
        self.img = get_image(tensor=self.img.cpu())
        # PIL photoimage does not implement zoom, so we use tkinter photoimage
        self.img = ImageTk.PhotoImage(image=self.img)
        # self.scale_img()
        self.label = Label(self.can1, image=self.img)
        self.label.image = self.img  # keep a reference!
        self.label.pack()

    def scale_img(self):
        scalew = self.displayed_image_size[0]/self.img.width()
        scaleh = self.displayed_image_size[1]/self.img.height()
        self.img.zoom(scalew, scaleh)

    def close_window(self, event):
        self.fen1.destroy()

    def navigate(self):
        # démarrage :
        self.fen1.mainloop()
