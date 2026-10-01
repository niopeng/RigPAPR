from .dataset import RINDataset
from torch.utils.data import DataLoader
import torch


def get_traindataset(args):
    return RINDataset(args, mode='train')


def get_trainloader(dataset, args):
    g = torch.Generator()
    g.manual_seed(torch.initial_seed())
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        generator=g,
    )


def get_testdataset(args):
    return RINDataset(args, mode='test')


def get_testloader(dataset, args):
    return DataLoader(dataset, batch_size=1, shuffle=False)


def get_dataset(args, mode):
    if mode == 'train':
        return get_traindataset(args)
    elif mode == 'test':
        return get_testdataset(args)
    else:
        raise ValueError("Unknown mode: {}".format(mode))


def get_loader(dataset, args, mode):
    if mode == 'train':
        return get_trainloader(dataset, args)
    elif mode == 'test':
        return get_testloader(dataset, args)
    else:
        raise ValueError("Unknown mode: {}".format(mode))
