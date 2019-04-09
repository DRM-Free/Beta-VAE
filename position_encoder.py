from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from tqdm import tqdm
import torch.nn.init as init


class position_encoder(nn.Module):
    def __init__(self, position_size, code_size):
        super(position_encoder, self).__init__()
        self.position_size = position_size
        self.code_size = code_size
        fully_connected_size = 50
        self.encoder = nn.Sequential(
            nn.Linear(self.position_size, fully_connected_size),
            torch.nn.LeakyReLU(),
            nn.Linear(self.fully_connected_size, fully_connected_size),
            torch.nn.LeakyReLU(),
            nn.Linear(self.fully_connected_size, fully_connected_size),
            torch.nn.LeakyReLU(),
            nn.Linear(self.fully_connected_size, self.code_size)
        )
        self.weight_init()

    def forward(self, position):
        return (self.encoder(position))

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                normal_init(m, 0, 1)


class position_code(dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        position = self.data_frame.iloc[idx, 1:].as_matrix()
        code = self.data_frame.iloc[idx, 2:].as_matrix()
        sample = {'position': position, 'code': code}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __call__(self, sample):
        position, code = sample['position'], sample['code']
        return {'position': torch.from_numpy(position),
                'code': torch.from_numpy(code)}


def normal_init(m, mean, std):
    m.weight.data.normal_(mean, std)
    if m.bias.data is not None:
        m.bias.data.zero_()


if __name__ == "__main__":
    csv_file = 'code_test.csv'
    root_dir = '.'
    dataset = position_code(csv_file, root_dir)
    dataloader = DataLoader(dataset, batch_size=4,
                            shuffle=True, num_workers=4)
    encoder = position_encoder(position_size=2, code_size=3)
    optimizer = torch.optim.Adadelta(params=position_encoder.parameters())

    n_epochs = 1000
    # number of times to repeat each batch (to reduce data transfert rate)
    n_repeat_batch = 10
    pbar = tqdm(total=n_epochs)
    for e in range(n_epochs):
        pbar.update(e)
        for i_batch, sample_batched in enumerate(dataloader):
            # print(i_batch, sample_batched['image'].size(),
            #       sample_batched['landmarks'].size())
            for repeat in range(n_repeat_batch):
                code = encoder.forward(sample_batched.position)
                optimizer.step()
                optimizer.zero_grad()
        pbar.update(1)
