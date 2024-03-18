import torch as th
import random
from diffclassification.diffusion import DDPM
from diffclassification.unet import MyUnet
from diffclassification.mnistdataset import get_train_data_loader
from diffclassification.utils import get_beta_schedule, train_model
import numpy as np
import argparse


def main(args):
    T = 1000
    BATCH_SIZE = 128
    LR = 0.001
    WEIGHT_DECAY = 0.0
    N_EPOCH = 30

    th.manual_seed(0)
    random.seed(0)

    model = MyUnet().to("cuda")

    ddpm = DDPM(betas=get_beta_schedule(T), model=model)
    dataloader = get_train_data_loader(args.path_to_mnist, batch_size=BATCH_SIZE, shuffle=True)

    train_model(
        ddpm=ddpm,
        dataloader=dataloader,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        n_epoch=N_EPOCH,
        device="cuda"
    )
    th.save(ddpm.to("cpu").state_dict(), args.path_to_save)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='MNIST train',
                    description='train diffusion on mnist dataset and save weights in path_to_save')
    parser.add_argument("path_to_mnist", type=str)
    parser.add_argument("path_to_save", type=str)
    args = parser.parse_args()
    main(args)