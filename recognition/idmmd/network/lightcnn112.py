import torch
import torch.nn as nn
import torch.nn.functional as F




class mfm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, type=1):
        super(mfm, self).__init__()
        self.out_channels = out_channels
        if type == 1:
            self.filter = nn.Conv2d(in_channels, 2 * out_channels, kernel_size=kernel_size, stride=stride,
                                    padding=padding)
        else:
            self.filter = nn.Linear(in_channels, 2 * out_channels)

    def forward(self, x):
        x = self.filter(x)
        out = torch.split(x, self.out_channels, 1)
        return torch.max(out[0], out[1])


class group(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(group, self).__init__()
        self.conv_a = mfm(in_channels, in_channels, 1, 1, 0)
        self.conv = mfm(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        x = self.conv_a(x)
        x = self.conv(x)
        return x


class resblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(resblock, self).__init__()
        self.conv1 = mfm(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = mfm(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + res
        return out



class network_29layers(nn.Module):
    def __init__(self, block, layers, num_classes=79077):
        super(network_29layers, self).__init__()
        self.conv1  = mfm(1, 48, 5, 1, 2)
        self.pool1  = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.block1 = self._make_layer(block, layers[0], 48, 48)
        self.group1 = group(48, 96, 3, 1, 1)
        self.pool2  = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.block2 = self._make_layer(block, layers[1], 96, 96)
        self.group2 = group(96, 192, 3, 1, 1)
        self.pool3  = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.block3 = self._make_layer(block, layers[2], 192, 192)
        self.group3 = group(192, 128, 3, 1, 1)
        self.block4 = self._make_layer(block, layers[3], 128, 128)
        self.group4 = group(128, 128, 3, 1, 1)
        self.pool4  = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.fc     = mfm(7*7*128, 256, type=0)
        self.fc2    = nn.Linear(256, num_classes)
            
    def _make_layer(self, block, num_blocks, in_channels, out_channels):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.block1(x)
        x = self.group1(x)
        x = self.pool2(x)

        x = self.block2(x)
        x = self.group2(x)
        x = self.pool3(x)

        x = self.block3(x)
        x = self.group3(x)
        x = self.block4(x)
        x = self.group4(x)
        x = self.pool4(x)

        x = x.view(x.size(0), -1)
        fc = self.fc(x)

        if self.training:
            x = F.dropout(fc, training=self.training)
            out = self.fc2(x)
            return out, F.normalize(fc,p=2,dim=1)
        return F.normalize(fc,p=2,dim=1)




################################
## cosface nets
################################

from torch.nn import Parameter


class network_29layers_cosface(nn.Module):
    def __init__(self, block, layers, num_classes=79077):
        super(network_29layers_cosface, self).__init__()
        self.conv1  = mfm(1, 48, 5, 1, 2)
        self.pool1  = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.block1 = self._make_layer(block, layers[0], 48, 48)
        self.group1 = group(48, 96, 3, 1, 1)
        self.pool2  = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.block2 = self._make_layer(block, layers[1], 96, 96)
        self.group2 = group(96, 192, 3, 1, 1)
        self.pool3  = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.block3 = self._make_layer(block, layers[2], 192, 192)
        self.group3 = group(192, 128, 3, 1, 1)
        self.block4 = self._make_layer(block, layers[3], 128, 128)
        self.group4 = group(128, 128, 3, 1, 1)
        self.pool4  = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.fc     = mfm(7*7*128, 256, type=0)

        self.weight = Parameter(torch.Tensor(num_classes, 256))
        nn.init.xavier_uniform_(self.weight)
    

    def _make_layer(self, block, num_blocks, in_channels, out_channels):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels, out_channels))
        return nn.Sequential(*layers)

    def cosine_sim(self, x1, x2, dim=1, eps=1e-8):
        ip = torch.mm(x1, x2.t())
        w1 = torch.norm(x1, 2, dim)
        w2 = torch.norm(x2, 2, dim)
        return ip / torch.ger(w1,w2).clamp(min=eps)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.block1(x)
        x = self.group1(x)
        x = self.pool2(x)

        x = self.block2(x)
        x = self.group2(x)
        x = self.pool3(x)

        x = self.block3(x)
        x = self.group3(x)
        x = self.block4(x)
        x = self.group4(x)
        x = self.pool4(x)

        x = x.view(x.size(0), -1)
        fc = self.fc(x)
    
        if self.training:
            x = F.dropout(fc, training=self.training)
            out = self.cosine_sim(x, self.weight)
            return out, F.normalize(fc, p=2, dim=1)
        return F.normalize(fc, p=2, dim=1)


def LightCNN_29Layers(**kwargs):
    model = network_29layers(resblock, [1, 2, 3, 4], **kwargs)
    return model

def LightCNN_29Layers_cosface(**kwargs):
    model = network_29layers_cosface(resblock, [1, 2, 3, 4], **kwargs)
    return model
