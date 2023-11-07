import torch.nn as nn
import torch.nn.functional as F
import math
import torch

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1,
                     bias=False)

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        # in_channels: channel of input channel attention
        super(ChannelAttention, self).__init__()
        self.avg = nn.AdaptiveAvgPool2d(output_size=1)
        self.max = nn.AdaptiveMaxPool2d(output_size=1)
        ## MLP
        self.fc1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // reduction, kernel_size=1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels=in_channels // reduction, out_channels=in_channels, kernel_size=1, bias=False)

        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max(x))))
        out = avg_out+max_out
        out = self.sigmod(out)
        return out


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        # in_channels: channel of input RCA
        # out_channels: channel of output RCA
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.ca = ChannelAttention(out_channels)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out1 = self.ca(out)
        out = out1 * out

        if self.downsample is not None:
            residual = self.downsample(x)
        out = out + residual
        return out

class SENet18(nn.Module):
    def __init__(self, num_classes):
        # num_classes: dimension of output feature
        self.in_channels = 64
        super(SENet18, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock, 64, blocks=2)
        self.layer2 = self._make_layer(BasicBlock, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, blocks=2, stride=2)
        self.Avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)
        #
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def _make_layer(self, BasicBlock, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != planes * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes * BasicBlock.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * BasicBlock.expansion))
        layers = []
        layers.append(BasicBlock(self.in_channels, planes, stride, downsample))
        self.in_channels = planes * BasicBlock.expansion
        for i in range(1, blocks):
            layers.append(BasicBlock(self.in_channels, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.bn1(x1)
        x3 = self.relu(x2)
        x4 = self.maxpool(x3)
        x5 = self.layer1(x4)
        x6 = self.layer2(x5)
        x7 = self.layer3(x6)
        x8 = self.layer4(x7)
        x9 = self.Avgpool(x8)
        x10 = x9.view(x9.size(0), -1)
        x11 = self.fc(x10)
        return x11

class double_conv2d_bn(nn.Module): #
    def __init__(self, in_channels, out_channels, kernel_size=3, strides=1, padding='same'):
        # in_channels: channel of input convolution in decoder
        # out_channels: channel of output convolution in decoder
        super(double_conv2d_bn,self).__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,
                                stride = strides,padding = padding ,bias =True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size = kernel_size,
                                stride = strides,padding = padding, bias = True)
        self.bn2 = nn.BatchNorm2d(out_channels)
    def forward(self,x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


class deconv2d_bn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, strides=2, padding=1):
        # in_channels: channel of input upsampling in decoder
        # out_channels: channel of output upsampling in decoder
        super(deconv2d_bn,self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels,out_channels,kernel_size=kernel_size,stride=strides,
                                        padding=padding,bias=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
    def forward(self,x):
        out = F.relu(self.bn1(self.conv1(x)))
        return out

class TCDHmodule(nn.Module):
    def __init__(self, num_classes,hash_bit):
        # num_classes: dimension of output feature
        # hash_bit: dimension of hash code
        super(TCDHmodule, self).__init__()
        self.encoder = SENet18(num_classes)
        self.u1 = double_conv2d_bn(512, 256)
        self.u2 = double_conv2d_bn(256, 128)
        self.u3 = double_conv2d_bn(128, 64)
        self.u4 = double_conv2d_bn(128, 64)
        self.u1 = double_conv2d_bn(512, 256)

        self.deconv0 = deconv2d_bn(512, 256)
        self.deconv1 = deconv2d_bn(256, 128)
        self.deconv2 = deconv2d_bn(128, 64)
        self.deconv3 = deconv2d_bn(64, 64)
        self.deconv4 = deconv2d_bn(64, 3)

        self.sigmoid = nn.Sigmoid()
        self.hash_layer = nn.Linear(num_classes, hash_bit)


    def forward(self, x):
        x1 = self.encoder.conv1(x)
        x2 = self.encoder.bn1(x1)
        x3 = self.encoder.relu(x2)
        x4 = self.encoder.maxpool(x3)
        x5 = self.encoder.layer1(x4)
        x6 = self.encoder.layer2(x5)
        x7 = self.encoder.layer3(x6)
        x8 = self.encoder.layer4(x7)
        x9 = self.encoder.Avgpool(x8)
        x10 = x9.view(x9.size(0), -1)
        embeddings = self.encoder.fc(x10)

        hash_code = self.hash_layer(embeddings)
        hash_like = self.sigmoid(hash_code)

        OO = self.deconv0(x8)
        c1 = torch.cat((OO, x7),dim=1)
        u1 = self.u1(c1)

        de1 = self.deconv1(u1)
        c2 = torch.cat((de1, x6), dim=1)
        u2 = self.u2(c2)

        de2 = self.deconv2(u2)
        c3 = torch.cat((de2, x5), dim=1)
        u3 = self.u3(c3)

        de3 = self.deconv3(u3)
        c4 = torch.cat((de3, x3), dim=1)
        u4 = self.u4(c4)

        de4 = self.deconv4(u4)
        decoder = self.sigmoid(de4)

        return decoder, embeddings, hash_code,  hash_like

