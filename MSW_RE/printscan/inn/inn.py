import torch
from torch import nn, Tensor
import torch.nn.functional as F


class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation, strides=1):
        super(Conv2D, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=strides, padding=1)
        self.activation = nn.ReLU() if activation == 'relu' else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class BaseFunc(nn.Module):
    def __init__(self, input_dim=2, output_dim=2):
        super(BaseFunc, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Encoder
        self.conv1 = Conv2D(input_dim, 8, 3, activation='relu', strides=2)
        self.conv2 = Conv2D(8, 16, 3, activation='relu', strides=2)
        self.conv3 = Conv2D(16, 32, 3, activation='relu', strides=2)

        # Decoder
        self.up1 = Conv2D(32, 16, 3, activation='relu')
        self.up2 = Conv2D(32, 8, 3, activation='relu')
        self.up3 = Conv2D(16, output_dim, 3, activation='relu')
        self.up_sample = nn.Upsample(scale_factor=(2, 2))

    def forward(self, img: Tensor):
        x1 = self.conv1(img)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)

        up1 = self.up1(self.up_sample(x3))
        up2 = self.up2(self.up_sample(torch.cat([x2, up1], dim=1)))
        output = self.up3(self.up_sample(torch.cat([x1, up2], dim=1)))
        return output


class Block_i(nn.Module):
    def __init__(self, input_dim=2, output_dim=2):
        super().__init__()
        self.ui = BaseFunc(input_dim, output_dim)
        self.qi = BaseFunc(input_dim, output_dim)

    def __call__(self, x1, x2, reverse=False):
        """

        :param reverse:
        :return:
        """
        if not reverse:
            y2 = x2 + self.ui(x1)
            y1 = x1 + self.qi(y2)
        else:
            y1 = self.qi(x2) - x1
            y2 = self.ui(y1) - x2
        return y1, y2


class INN(nn.Module):
    def __init__(self, input_dim=2, output_dim=2):
        super().__init__()
        self.block1 = Block_i(input_dim, output_dim)
        self.block2 = Block_i(input_dim, output_dim)
        self.block3 = Block_i(input_dim, output_dim)
        self.block4 = Block_i(input_dim, output_dim)
        self.block5 = Block_i(input_dim, output_dim)
        self.block6 = Block_i(input_dim, output_dim)
        self.block7 = Block_i(input_dim, output_dim)
        self.block8 = Block_i(input_dim, output_dim)
        self.block9 = Block_i(input_dim, output_dim)
        self.block10 = Block_i(input_dim, output_dim)

    def __call__(self, x1, x2, reverse=False):
        if not reverse:
            x1, x2 = self.block1(x1, x2)
            x1, x2 = self.block2(x1, x2)
            x1, x2 = self.block3(x1, x2)
            x1, x2 = self.block4(x1, x2)
            x1, x2 = self.block5(x1, x2)
            x1, x2 = self.block6(x1, x2)
            x1, x2 = self.block7(x1, x2)
            x1, x2 = self.block8(x1, x2)
            x1, x2 = self.block9(x1, x2)
            out1, out2 = self.block10(x1, x2)
            return out1, out2
        else:
            x1, x2 = self.block10(x1, x2)
            x1, x2 = self.block9(x1, x2)
            x1, x2 = self.block8(x1, x2)
            x1, x2 = self.block7(x1, x2)
            x1, x2 = self.block6(x1, x2)
            x1, x2 = self.block5(x1, x2)
            x1, x2 = self.block4(x1, x2)
            x1, x2 = self.block3(x1, x2)
            x1, x2 = self.block2(x1, x2)
            out1, out2 = self.block1(x1, x2)
            return out1, out2


class HaarDownsampling(nn.Module):
    def __init__(self, channel_in):
        super(HaarDownsampling, self).__init__()
        self.channel_in = channel_in

        self.haar_weights = torch.ones(4, 1, 2, 2)

        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1

        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1

        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1

        self.haar_weights = torch.cat([self.haar_weights] * self.channel_in, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = False

    def forward(self, x, rev=False):
        if not rev:
            out = F.conv2d(x, self.haar_weights, bias=None, stride=2, groups=self.channel_in) / 4.0
            out = out.reshape([x.shape[0], self.channel_in, 4, x.shape[2] // 2, x.shape[3] // 2])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2] // 2, x.shape[3] // 2])
            out1 = out[:, 0:out.shape[1] // 2, :, :]
            out2 = out[:, out.shape[1] // 2:, :, :]
            return out1, out2

        else:
            x_cat = torch.cat((x[0], x[1]), 1)
            out = x_cat.reshape([x_cat.shape[0], 4, self.channel_in, x_cat.shape[2], x_cat.shape[3]])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x_cat.shape[0], self.channel_in * 4, x_cat.shape[2], x_cat.shape[3]])
            return F.conv_transpose2d(out, self.haar_weights, bias=None, stride=2, groups=self.channel_in)


class INNChannel(nn.Module):
    def __init__(self, input_dim=1, l: int = 6):
        super().__init__()
        self.l = l
        self.dwt = HaarDownsampling(input_dim)
        self.inn = INN()

    def __call__(self, img, reverse=False):
        """

        :param img:
        :return:
        """
        x1, x2 = self.dwt(img, rev=False)
        out1, out2 = self.inn(x1, x2, reverse)
        out = self.dwt([out1, out2], rev=True)
        return torch.sigmoid(out)


#
inn = INNChannel()
x = torch.randint(0, 256, size=(1, 1, 256, 256)).float()
out = inn(x, False)
out_round = torch.round(out * 255)
out_ = inn(out_round, True)
print(torch.sum(out_ - x))
print(out.shape)
