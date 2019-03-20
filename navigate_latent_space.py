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


class latent_space_navigator(object):
    def __init__(self, sol):
        self.current_navigated_dim = 0
        self.navigation_step = 0.1
        self.displayed_image_size = [200, 200]
        self.fen1 = Tk()
        self.sol = sol
        self.latent_position=[0]*self.sol.z_dim
        # création de widgets Label()
        Labels_frame = Frame(self.fen1)
        Labels_frame.grid(sticky=E, row=0, column=0)
        Label(Labels_frame, text='Controls :').pack()
        Label(Labels_frame, text='Change exploration step : +-').pack()
        Label(Labels_frame,
              text='Change explored latent dimension : up down arrrows').pack()
        Label(Labels_frame, text='Explore dimension : left right arrows').pack()

        Label(Labels_frame, text='Info:').pack()
        # Next labels need updating so give them a name and keep them in self
        self.info_dim = Label(Labels_frame, text='Currently explored dimension : {0}'.format(
            self.current_navigated_dim))
        self.info_dim.pack()
        self.info_step = Label(Labels_frame, text='Current step value :{0}'.format(
            self.navigation_step))
        self.info_step.pack()
        self.info_latent_pos = Label(Labels_frame, text='Current latent position :{0}'.format(
            self.latent_position))
        self.info_step.pack()

        # création d'un widget 'Canvas' contenant une image bitmap :
        # self.can1 = Canvas(
        # self.fen1, width=self.displayed_image_size[0], height=self.displayed_image_size[1], bg='white')
        # sol.net_mode(train=False)

        # self.can1.grid(row=0, column=2, rowspan=4, padx=10, pady=5)
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

        # self.latent_position = torch.tensor(
        #     [random.uniform(-2, 2)]*self.sol.z_dim).cuda()
        # self.img_arr = F.sigmoid(
        #     self.sol.net.decoder(self.latent_position)).data
        # self.img_arr = get_image(tensor=self.img_arr.cpu())

        # testing update with original images
        ind = randint(1, self.sol.data_loader.dataset.__len__())
        self.img_arr = self.sol.data_loader.dataset.__getitem__(ind)
        self.img_arr = Variable(
            cuda(self.img_arr, self.sol.use_cuda), volatile=True).unsqueeze(0)
        self.img_arr = get_image(tensor=self.img_arr.cpu())
        # testing code end

        self.scale_img()
        img = ImageTk.PhotoImage(image=self.img_arr)
        self.label.configure(image=img)
        # self.label.update()
        # self.label.update_idletasks()
        # self.fen1.update_idletasks()

    def init_img(self):
        self.img_arr = self.sol.data_loader.dataset.__getitem__(0)
        self.img_arr = Variable(
            cuda(self.img_arr, self.sol.use_cuda), volatile=True).unsqueeze(0)
        self.img_arr = get_image(tensor=self.img_arr.cpu())
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
