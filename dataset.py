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
from random import sample
from operator import itemgetter


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
            cam_pos = self.cam_pos[index]
            dict_pos = {"x": cam_pos[0], "y": cam_pos[1], "z": cam_pos[2]}
        return img, dict_pos

    def get_pos(self, index):
        return self.cam_pos[index]

    def __len__(self):
        return len(self.imgs)


# This secondary dataset is meant for paired ambiguity retreival and training of VAE encoder and Auxiliary Network (AN)
# Notice : for ambiguity computation, images code should be available (call compute_codes first)
class In_memory_paired_dataset:
    # Actually pairwise ambiguity is computed only for a number of pairs equal to batch size (one pair per batch).
    # FOr further optimization, keeping track of already computed ambiguity pairs and training several times with the same
    # ambiguity data could be done
    # type : "albiguity" or "similarity"
    def __init__(self, image_dataset, type, pool_size=50, transform=None):
        self.transform = transform
        self.type = type
        self.pairs, self.pairs_nb = make_pairs(pool_size)
        self.used_pairs = self.pairs
        self.used_pairs_nb = self.pairs_nb
        self.image_dataset = image_dataset
        self.pool_size = pool_size
        self.already_passed = False
        self.sample_images()

    def sample_images(self):
        self.used_pairs = self.pairs
        self.used_pairs_nb = self.pairs_nb
        self.pick_images()
        self.compute_pairwise_ambiguities()

    def pick_images(self):
        # Generate n distinct integers within image dataset
        self.imgs = [None]*self.pool_size
        img_indexes = sample(
            range(0, self.image_dataset.__len__()), self.pool_size)
        self.imgs = list(itemgetter(*img_indexes)(self.image_dataset.imgs))
        # img_indexes[i] is the corresponding index in the image dataset for an image of index i in self.imgs of the pairwise dataset
        self.img_indexes = img_indexes

    def __getitem__(self, index):
        # Return pair of images indexes and supervised ambiguity expectation
        # img_indexes = self.pairs[self.img_indexes[index]] # can not extract images from dataloader this way
        # TODO maybe remove self.imgs, as images are also stored in the image dataset
        # im1 = self. imgs[self.pairs[index][0]] #those images are untransformed, thus not in tensor form
        # im2 = self.imgs[self.pairs[index][1]]
        # TODO fix getitem to make images pair and supervised information available at the same time
        im1, pos = self.image_dataset.__getitem__(
            self.img_indexes[self.used_pairs[index][0]])
        im2, pos = self.image_dataset.__getitem__(
            self.img_indexes[self.used_pairs[index][1]])
        # TODO check concatentation axis, check items are returned as tensors when required
        # ambiguity = torch.FloatTensor(
        # self.ambiguities[index])
        if self.type == "ambiguity":
            ambiguity = self.ambiguities[index]
            data = {"im1": im1, "im2": im2, "ambiguity": ambiguity}
        if self.type == "similarity":
            similarity = self.similarities[index]
            data = {"im1": im1, "im2": im2, "similarity": similarity}

        # self.ambiguities[index], dtype = torch.float64, requires_grad = True)
        return data

    # def get_ambiguity(self, index):
    #     return self.transform(self.ambiguities[index])

    def __len__(self):
        return self.used_pairs_nb
        # TODO keep only ambiguities superior to a threshold (0 is fair) and the same number of non ambiguous pairs
        # Approx 7% of pairs are ambiguous wrt mean squares : too much negative examples is not good

    def compute_pairwise_ambiguities(self):
        # Debug code
        if self.already_passed == False:
            self.already_passed = True
        else:
            a = 1
# Debug code

        pairs_nb = self.pairs_nb
        # img_MI = [0] * pairs_nb
        img_errors = [0] * pairs_nb
        cam_errors = [0] * pairs_nb
        for index in range(self.pairs_nb):
            position_index_1 = self.img_indexes[self.pairs[index][0]]
            position_index_2 = self.img_indexes[self.pairs[index][1]]
            cam_pos1 = self.image_dataset.get_pos(position_index_1)
            img1 = self.imgs[self.pairs[index][0]]
            cam_pos2 = self.image_dataset.get_pos(position_index_2)
            img2 = self.imgs[self.pairs[index][1]]
            img1 = np.array(img1)
            img1 = np.reshape(img1, -1)
            img2 = np.array(img2)
            img2 = np.reshape(img2, -1)
            img_errors[index] = mean_squared_error(img1, img2)
            # img_MI[index] = mutual_info_score(img1, img2)
            cam_errors[index] = mean_squared_error(cam_pos1, cam_pos2)
        img_errors = scale(img_errors, axis=0, with_mean=True,
                           with_std=True, copy=True)
        # img_MI = scale(img_MI, axis=0, with_mean=True,
        #                with_std=True, copy=True)
        cam_errors = scale(cam_errors, axis=0, with_mean=True,
                           with_std=True, copy=True)

        # ambiguity, to be maximized by Adversarial Net and minimized by VAE
        # ambiguities = (img_MI + img_errors) / 2 - cam_errors
        # similarities = cam_errors + (img_MI - img_errors) /scale 2

        if self.type == "ambiguity":
            # Find the 5% most ambiguous and 5% least ambiguous and give them an ambiguity class {0,1}
            ambiguities = img_errors - cam_errors
            keep = int(np.ceil(np.size(ambiguities) * 0.1))
            amb_indexes = np.argsort(ambiguities)
            ambiguous_pairs = amb_indexes[:keep]
            not_ambiguous_pairs = amb_indexes[-keep:]
            self.used_pairs = np.concatenate((
                [self.pairs[i] for i in ambiguous_pairs], [self.pairs[i] for i in not_ambiguous_pairs]), axis=0)
            self.used_pairs_nb = self.used_pairs.__len__()
            self.ambiguities = np.concatenate(([0]*keep, [1]*keep))

        if self.type == "similarity":
            # Find the n% most similar and n% least similar and give them a similarity class {0,1}
            similarities = -1 * cam_errors
            keep = int(np.ceil(np.size(similarities) * 0.2))
            sim_indexes = np.argsort(similarities)
            similar_pairs = sim_indexes[:keep]
            not_similar_pairs = sim_indexes[-keep:]
            self.used_pairs = np.concatenate((
                [self.pairs[i] for i in similar_pairs], [self.pairs[i] for i in not_similar_pairs]), axis=0)
            self.used_pairs_nb = self.used_pairs.__len__()
            self.similarities = np.concatenate(([0] * keep, [1] * keep))


def get_image_dataloader(args):
    name = args.dataset
    dset_dir = args.dset_dir
    batch_size = args.batch_size
    num_workers = int(np.floor(args.num_workers/2))
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
                              #   num_workers=num_workers,
                              num_workers=0,
                              pin_memory=True,
                              drop_last=True)
    data_loader = train_loader
    return data_loader, train_data

# TODO verify that shuffling of both dataloaders does not mismatch supervised training data


def get_pairwise_dataloader(im_dataset, args, type):
    transform = transforms.ToTensor()
    dset = In_memory_paired_dataset(
        im_dataset, type, transform=transform, pool_size=50)
    return DataLoader(dset,
                      batch_size=args.batch_size,
                      shuffle=True,
                      num_workers=0,
                      #   num_workers=int(np.floor(args.num_workers/2)),
                      pin_memory=True,
                      drop_last=True)
