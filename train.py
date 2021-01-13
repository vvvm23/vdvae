import torch
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image

import numpy as np
from tqdm import tqdm

import itertools

from hps import HPS
from helper import info, error, warning
from helper import get_device
from vae import VAE

def load_dataset(dataset, batch_size):
    if dataset == 'cifar10':
        # TODO: Normalize input images
        # TODO: Other transforms too
        train_dataset = torchvision.datasets.CIFAR10('data', train=True, download=True, transform=transforms.ToTensor())
        test_dataset = torchvision.datasets.CIFAR10('data', train=False, download=True, transform=transforms.ToTensor())
    else:
        error(f"Unrecognized dataset '{dataset}'! Exiting..")
        exit()

    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

    # return train_loader, test_loader
    return test_loader, test_loader

def train(model, loader, optim, crit, device):
    total_loss, r_loss, kl_loss = 0.0, 0.0, 0.0
    model.train()
    for x, _ in tqdm(loader):
        optim.zero_grad()
        x = x.to(device)

        y, decoder_kl = model(x)
        rl = crit(x, y).mean(dim=(1,2,3))

        rpp = torch.zeros_like(rl)
        for k in decoder_kl:
            rpp += k.sum(dim=(1,2,3))
        rpp /= np.prod(x.shape[1:])
        
        elbo = (rpp*0.25 + rl).mean()
        elbo.backward()
        optim.step()

        rl = rl.mean()
        kl = rpp.mean()

        total_loss += elbo
        r_loss += rl
        kl_loss += kl
    return total_loss / len(loader), r_loss / len(loader), kl_loss / len(loader)

def evaluate(model, loader, optim, crit, device, img_id=None):
    total_loss, r_loss, kl_loss = 0.0, 0.0, 0.0
    with torch.no_grad():
        model.eval()
        for i, (x, _) in enumerate(tqdm(loader)):
            optim.zero_grad()
            x = x.to(device)

            y, decoder_kl = model(x)
            rl = crit(x, y).mean(dim=(1,2,3))

            rpp = torch.zeros_like(rl)
            for k in decoder_kl:
                rpp += k.sum(dim=(1,2,3))
            rpp /= np.prod(x.shape[1:])
            
            elbo = (rpp*0.25 + rl).mean()
            rl = rl.mean()
            kl = rpp.mean()

            total_loss += elbo
            r_loss += rl
            kl_loss += kl

            if img_id != None and i == 0:
                save_image(y, f"imgs/eval-recon-{img_id}.png")

        return total_loss / len(loader), r_loss / len(loader), kl_loss / len(loader)

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    device = get_device(HPS.cuda)
    train_loader, test_loader = load_dataset(HPS.dataset, HPS.batch_size)

    model = VAE(HPS.in_channels, HPS.h_width, HPS.m_width, HPS.z_dim, 
                nb_blocks=HPS.nb_blocks, nb_res_blocks=HPS.nb_res_blocks, scale_rate=HPS.scale_rate).to(device)
    info(f"Number of trainable parameters {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    optim = torch.optim.Adam(model.parameters(), lr=HPS.lr, weight_decay=HPS.decay)
    crit = torch.nn.MSELoss(reduction='none')

    nb_iterations = 0
    # for ei in range(HPS.nb_epochs):
    for ei in itertools.count():
        train_loss, r_loss, kl_loss = train(model, train_loader, optim, crit, device)
        nb_iterations += len(train_loader)

        info(f"training, epoch {ei+1} \t iter: {nb_iterations} \t loss: {train_loss} \t r_loss {r_loss} \t kl_loss {kl_loss}")
        eval_loss, r_loss, kl_loss = evaluate(model, test_loader, optim, crit, device, img_id=ei)
        info(f"evaluate, epoch {ei+1}/{HPS.nb_epochs} \t iter: {nb_iterations} \t loss: {eval_loss} \t r_loss {r_loss} \t kl_loss {kl_loss}")

        if nb_iterations > HPS.nb_iterations:
            break
