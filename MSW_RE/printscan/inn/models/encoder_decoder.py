import torch
import torch.nn as nn
from printscan.inn.models.Inn import Noise_INN, INN
from printscan.inn.block.Haar import HaarDownsampling


class INL(nn.Module):
    def __init__(self):
        super(INL, self).__init__()
        self.model = Noise_INN()
        self.haar = HaarDownsampling(3)

    def forward(self, x, reverse=False):

        if not reverse:
            out = self.haar(x)
            out = self.model(out)
            out = self.haar(out, rev=True)

        else:
            out = self.haar(x)
            out = self.model(out, rev=True)
            out = self.haar(out, rev=True)

        return out


class FED(nn.Module):
    def __init__(self, diff=False, length=64):
        super(FED, self).__init__()
        self.model = INN(diff, length)

    def forward(self, x, rev=False):

        if not rev:
            out = self.model(x)

        else:
            out = self.model(x, rev=True)

        return out


# #
# inn = INL()
# x = torch.rand(size=(1, 3, 128, 128))
# out = inn(x, True)
# print(out.shape)

x = torch.randint(0, 2, (1, 3, 256, 256)).float()
ninn = INL()
y = ninn(x)
out = ninn(y, True)
print(torch.mean(x - out))
