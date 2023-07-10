import torch
torch.pi = torch.acos(torch.zeros(1)).item() * 2
import torch.nn.functional as F
from torch import nn
import logging
import argparse
import numpy as np
from torch.optim.lr_scheduler import StepLR

from tqdm import trange

# %%
import modules.functions_hv_grad_3d

# %%
from modules.functions_evaluation import compute_hv_in_higher_dimensions as compute_hv

# %%
import numpy as np
import torch

from modules.functions_evaluation import fastNonDominatedSort
from modules.functions_hv_grad_3d import grad_multi_sweep_with_duplicate_handling

from model.NCF import NCF_Hyper,NCF_Target 

from modules.functions_evaluation import fastNonDominatedSort

from loss.ncf_loss import NCFLoss
#from loss.FactorVae_loss import FactorVAELoss
from data_loader.amazonDataset import AmazonDataset
from torch.utils.data import DataLoader
import os
from torch.autograd import Variable





def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        torch.nn.init.normal_(m.bias.data, 0.0, 0.001)

# %%


# %%
# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)
def count_parameters(model, requires_grad=True):
    return sum(p.numel() for p in model.parameters() if p.requires_grad == requires_grad)

import random
def set_logger():
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )


def set_seed(seed):
    """for reproducibility
    :param seed:
    :return:
    """
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


set_seed(0)
# %%
set_logger()

class parse_arg():
  def __init__(self):
    pass
args = parse_arg()
args.no_cuda = False
args.gpus = '2'

def get_device(no_cuda=False, gpus='0'):
    return torch.device(f"cuda:{gpus}" if torch.cuda.is_available() and not no_cuda else "cpu")


def train(device: torch.device, lr: float, wd: float, bs: int, epochs:int,
          num_ray:int, drop_out
          ):
    
    hnet: nn.Module = NCF_Hyper(drop_out=drop_out)
    net: nn.Module = NCF_Target()

    logging.info(f"HN size: {count_parameters(hnet)}")
    logging.info(f"TN size: {count_parameters(net, False)}")

    hnet = hnet.to(device)
    net = net.to(device)

    train_path = "/home/ubuntu/duc.nm195858/Hypernet-NCF/data/training_dataset.npy"
    val_path  = "/home/ubuntu/duc.nm195858/Hypernet-NCF/data/val_dataset.npy"
    test_path = "/home/ubuntu/duc.nm195858/Hypernet-NCF/data/test_dataset.npy"

    train_dataset = AmazonDataset(train_path)
    val_dataset = AmazonDataset(val_path)
    test_dataset = AmazonDataset(test_path) 

    train_loader = DataLoader(train_dataset, batch_size = bs, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = bs, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle= True)
    
    loss_1 = NCFLoss()
    loss_2 = NCFLoss()

    best_hv = 0
    patience = 0
    early_stop = 0
    training_loss = []

    val_hv = []
    val_loss = []
    test_loss = []

    optimizer = torch.optim.Adam(hnet.parameters(), lr=lr, weight_decay=wd)

    patience = 0
    epoch_iter = trange(epochs)
    for epoch in epoch_iter:
        if (patience+1) % 25 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= np.sqrt(0.5)
            patience = 0
            lr *= np.sqrt(0.5)
            #hesophat = max(1, hesophat*0.85)
            print('Reduce the learning rate {}'.format(lr))
        training_loss_epoch = []
        for i, batch in enumerate(train_loader):
            hnet.train()

            users = batch[0]
            users = Variable(users).to(device)
            items = batch[1]
            items = Variable(items).to(device)
            labels = batch[2]
            labels = Variable(labels).to(device)
            price = batch[3]
            price = Variable(price).to(device)

            start = 0
            end = np.pi/2
            random = np.random.uniform(start, end)
            ray = np.array([np.cos(random),
                                np.sin(random)], dtype='float32')

            ray /= ray.sum()
            ray = torch.from_numpy(
                                ray.flatten()
                            ).to(device)

            weights = hnet(ray)
            model_output = net(users, items , weights)

            l1 = loss1.compute_loss(labels, model_output)/250
            l2 = loss2.compute_loss(labels, model_output, price)/2500

            loss = ray[0]*l1 + ray[1]*l2 
            losses_numpy = [l1.cpu().detach().numpy(), l2.cpu().detach().numpy()]
            training_loss_epoch.append(losses_numpy)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        training_loss_epoch = np.mean(np.stack(training_loss_epoch), axis=0)
        training_loss.append(training_loss_epoch)

        print(f'Epoch {epoch}: ')
        print(f"Loss 1{training_loss_epoch[0]}, loss_2(revenue): {training_loss_epoch[1]}")

if __name__ == "__main__":

    device = get_device(no_cuda=args.no_cuda, gpus=args.gpus)
    wd=0.
    epochs=1000
    num_ray = 25
    lr = 5e-4
    bs = 500
    drop_out = 0.15
    train(device = device, lr=lr, wd=wd, epochs=epochs, bs=bs, num_ray=num_ray, drop_out =drop_out)






