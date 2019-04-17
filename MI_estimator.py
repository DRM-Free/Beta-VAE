from torch import nn, optim, reshape, Size as tsize, mean as tmean, log, exp, save, tensor, cat
from dataset import get_image_dataloader, get_pairwise_dataloader
from tqdm import tqdm
from numpy import random, array, arange
import holoviews as hv
from model import kaiming_init


class MI_estimator(nn.Module):
    def __init__(self, dim):
        super(MI_estimator, self).__init__()
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(2*dim, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 1))
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x, y):
        return self.net(cat((x, y)))


class minimal_args:
    def __init__(self):
        self.dataset = "cube_small"
        self.dset_dir = "data"
        self.batch_size = 100
        self.num_workers = 0
        self.image_size = 64


def save_state(net, optim):
    model_states = {'VAE_net': net.state_dict()}
    optim_states = {'VAE_optim': optim.state_dict()}
    states = {'model_states': model_states,
              'optim_states': optim_states}

    file_path = "MI_estimator"
    with open(file_path, mode='wb+') as f:
        save(states, f)


H = 10
n_epoch = 1
data_size = 20000
dim = 64*3
model = MI_estimator(dim=dim)
optimizer = optim.Adadelta(model.parameters(), lr=0.01)
plot_loss = []
args = minimal_args()
image_dataloader, im_dataset = get_image_dataloader(args)
pairwise_dataloader = get_pairwise_dataloader(im_dataset, args)
pbar = tqdm(total=n_epoch)

# Get element total size
data = pairwise_dataloader.dataset.__getitem__(0)
imgs, ambiguity, similarity = data
x_sample = imgs[0]
x_sample = reshape(x_sample, (-1,))
tot_size = x_sample.size()[0]*args.batch_size

for i in range(n_epoch):
    for _, data in enumerate(pairwise_dataloader, 0):
        # data contains the training data for a whole batch. Beware of dimensions mismatch after training !
        imgs, ambiguity, similarity = data
        x_sample = imgs[0]
        x_sample = reshape(x_sample, (-1,))
        y_sample = imgs[1]
        y_sample = reshape(y_sample, (-1,))
        order = array(range(tot_size))
        random.shuffle(order)

        # in-place changing of values
        y_sample[array(range(tot_size))] = y_sample[order]
        y_shuffle = random.permutation(y_sample)
        y_shuffle = tensor(y_shuffle)
        y_shuffle = y_shuffle.cuda()
        x_sample = x_sample.cuda()
        y_sample = y_sample.cuda()
        x_sample.requires_grad = True
        y_sample.requires_grad = True
        y_shuffle.requires_grad = True

        pred_xy = model.forward(x_sample, y_sample)
        pred_x_y = model.forward(x_sample, y_shuffle)

        ret = tmean(pred_xy) - log(tmean(exp(pred_x_y)))
        loss = - ret  # maximize
        plot_loss.append(loss.data.numpy())
        pbar.write("loss={}".format(loss))
        model.zero_grad()
        loss.backward()
        optimizer.step()
    pbar.update(1)
    pairwise_dataloader.dataset.pick_images

save_state(model, optimizer)
plot_x = arange(len(plot_loss))
plot_y = array(plot_loss).reshape(-1,)
hv.Curve((plot_x, -plot_y)) * hv.Curve((plot_x, mi))
