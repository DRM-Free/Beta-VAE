"""dataset.py"""

import os
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import pandas as pd
from create_ambiguity_matrix import make_pairs
import csv
from create_ambiguity_matrix import get_cam_pos, get_ambiguities
from sklearn.metrics import mean_squared_error, mutual_info_score
from sklearn.preprocessing import scale
import cv2
from tqdm import tqdm
from PIL import Image


def is_power_of_2(num):
    return ((num & (num - 1)) == 0) and num != 0


# class CustomImageFolder(ImageFolder):
#     def __init__(self, root, transform=None):
#         super(CustomImageFolder, self).__init__(root, transform)

#     def __getitem__(self, index):
#         path = self.imgs[index][0]
#         img = self.loader(path)
#         if self.transform is not None:
#             img = self.transform(img)
#         return img

# This dataset class replaces CustomImageFolder
# TODO test other color spaces some time
class In_memory_dataset:

    def __init__(self, dset_dir, file, transform=None):
        # self.imgs = load_images_from_folder(root) #would not guarantee images are in the proper order
        image_folder = os.path.join(dset_dir, file, "images")
        img_number = len(os.listdir(image_folder))
        self.imgs = [None] * img_number
        # Ensure images are loaded in proper order
        pbar = tqdm(total=img_number)
        print("loading the dataset in memory (", img_number, ") images")
        for i in range(img_number):
            img_name = "viewpoint" + str(i) + ".png"
            # self.imgs[i] = cv2.imread(os.path.join(image_folder, img_name))
            self.imgs[i] = Image.open(os.path.join(image_folder, img_name))
            pbar.update(1)
        pbar.close()
        self.transform = transform
        # Get euclidean camera positions (list of [x,y,z]), TODO check that the file is correctly found
        self.cam_pos = get_cam_pos(
            os.path.join(dset_dir, "camera_data_"+file + ".csv"), len(self.imgs))

    def __getitem__(self, index):
        img = self.imgs[index]
        if self.transform is not None:
            img = self.transform(img)
        return img
# TODO replace by a function that gives position only. image is already provided by getitem

    def get_pos(self, index):
        return self.cam_pos[index]

    def __len__(self):
        return len(self.imgs)


# This secondary dataset is meant for paired ambiguity retreival and training of VAE encoder and Auxiliary Network (AN)
# Notice : for ambiguity computation, images code should be available (call compute_codes first)
class In_memory_paired_dataset:
    # pos_file_name = "camera_data_cube_64_R_4_random_angles.csv"
    # TODO make sure only part of the full dataset is given to paired dataset to prevent high number of pairs (recommended 100 elements)
    # Actually pairwise ambiguity is computed only for a number of pairs equal to batch size (one pair per batch).
    # FOr further optimization, keeping track of already computed ambiguity pairs and training several times with the same
    # ambiguity data could be done

    def __init__(self, image_dataset, pos_file_name, pool_size=100):
        self.pairs, self.pairs_nb = make_pairs(pool_size)
        self.image_dataset = image_dataset

    def __getitem__(self, index):
        concatenated_input = np.concatenate((
            self.image_dataset.get_pos(self.pairs[index][0]), self.image_dataset.get_pos(self.pairs[index][1])), axis=0)
        return concatenated_input

    def __len__(self):
        return self.pairs_nb

    def compute_codes(self):
        pass

    def compute_pairwise_ambiguities(self):
        pairs_nb = self.pairs_nb
        img_MI = [0] * pairs_nb
        img_errors = [0] * pairs_nb
        cam_errors = [0] * pairs_nb
        for index in range(self.pairs_nb):
            cam_pos1 = self.image_dataset.get_pos(
                self.pairs[index][0])
            img1 = self.image_dataset.__getitem__(self.pairs[index][0])
            cam_pos2 = self.image_dataset.get_pos(
                self.pairs[index][1])
            img2 = self.image_dataset.__getitem__(self.pairs[index][1])
            img_errors[index] = mean_squared_error(img1, img2)
            img_MI[index] = mutual_info_score(
                np.reshape(img1, -1), np.reshape(img2, -1))
            cam_errors[index] = mean_squared_error(cam_pos1, cam_pos2)
        img_errors = scale(img_errors, axis=0, with_mean=True,
                           with_std=True, copy=True)
        img_MI = scale(img_MI, axis=0, with_mean=True,
                       with_std=True, copy=True)
        cam_errors = scale(cam_errors, axis=0, with_mean=True,
                           with_std=True, copy=True)
        ambiguities = cam_errors + (img_MI - img_errors) / 2
        self.ambiguities = ambiguities


def get_image_dataloader(args):
    name = args.dataset
    dset_dir = args.dset_dir
    batch_size = args.batch_size
    num_workers = args.num_workers
    image_size = args.image_size
    assert image_size == 64, 'currently only image size of 64 is supported'

    if name.lower() == 'cube_small':
        file = 'cube_64_R_4_random_angles'
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(), ])
        train_kwargs = {'dset_dir': dset_dir,
                        'file': file, 'transform': transform}
        dset = In_memory_dataset

    elif name.lower() == 'cube_random_spherical_position':
        file = 'cube_64_random_spherical_position'
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(), ])
        train_kwargs = {'dset_dir': dset_dir,
                        'file': file, 'transform': transform}
        dset = In_memory_dataset

    elif name.lower() == 'two_balls':
        file = 'two_balls'
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(), ])
        train_kwargs = {'dset_dir': dset_dir,
                        'file': file, 'transform': transform}
        dset = In_memory_dataset

    else:
        raise NotImplementedError

    train_data = dset(**train_kwargs)
    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True,
                              drop_last=True)
    data_loader = train_loader
    return data_loader

# TODO verify dataloader


def get_pairwise_dataloader(args):
    dset = In_memory_paired_dataset
    return DataLoader(dset,
                      batch_size=args.batch_size,
                      shuffle=True,
                      num_workers=args.num_workers,
                      pin_memory=True,
                      drop_last=True)
