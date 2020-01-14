import torch
from torch import nn
from torch.autograd import Variable
from constantes import *


class NN(nn.Module):

    def __init__(self):
        super(NN, self).__init__()
        self.inp = nn.Linear(4, N_IN)
        self.int = nn.Linear(N_IN, N_INTER)
        self.out = nn.Linear(N_INTER, 2)

    def forward(self, ob):
        ob.requires_grad = True
        return self.out(nn.functional.relu(self.int(nn.functional.relu(self.inp(ob)))))

