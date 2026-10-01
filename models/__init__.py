import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import lpips
from math import exp
from .model import PAPR
from .loss import get_loss


def get_model(args, device='cuda'):
    return PAPR(args, device=device)
