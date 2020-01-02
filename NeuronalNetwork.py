import torch
from torch import nn
from torch.autograd import Variable


class NN(nn.Module):

    def __init__(self, N_in):
        super(NN, self).__init__()
        self.inp = nn.Linear(4, N_in)
        self.out = nn.Linear(N_in, 2)

    def forward(self, ob):
        ob = torch.from_numpy(ob)
        ob.requires_grad = True
        return self.out(nn.functional.relu(self.inp(ob)))

