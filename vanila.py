# image segmentaion with dice + BCE loss deep learning model
# test the model with random input
from torch import nn
from torchvision import models
from operations import MixedOp
import torch


ranges = {"vgg16": ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31))}


class VGGNet(nn.Module):
    def __init__(self, pretrained=True):
        super(VGGNet, self).__init__()
        self.ranges = ranges["vgg16"]
        self.features = models.vgg16(weights=pretrained).features

    def forward(self, x):
        output = {}
        for idx in range(len(self.ranges)):
            for layer in range(self.ranges[idx][0], self.ranges[idx][1]):
                x = self.features[layer](x)
            output["x%d" % (idx + 1)] = x
        return output


class FCNs(nn.Module):
    def __init__(self, pretrained_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(
            512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1
        )
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(
            512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1
        )
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(
            256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1
        )
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(
            128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1
        )
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(
            64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1
        )
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        output = self.pretrained_net(x)

        x5 = output["x5"]  # size=(N, 512, x.H/32, x.W/32)
        x4 = output["x4"]  # size=(N, 512, x.H/16, x.W/16)
        x3 = output["x3"]  # size=(N, 256, x.H/8,  x.W/8)
        x2 = output["x2"]  # size=(N, 128, x.H/4,  x.W/4)
        x1 = output["x1"]  # size=(N, 64, x.H/2,  x.W/2)

        score = self.bn1(self.relu(self.deconv1(x5)))  # size=(N, 512, x.H/16, x.W/16)
        score = score + x4  # element-wise add, size=(N, 512, x.H/16, x.W/16)
        score = self.bn2(self.relu(self.deconv2(score)))  # size=(N, 256, x.H/8, x.W/8)
        score = score + x3  # element-wise add, size=(N, 256, x.H/8, x.W/8)
        score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
        score = score + x2  # element-wise add, size=(N, 128, x.H/4, x.W/4)
        score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)
        score = score + x1  # element-wise add, size=(N, 64, x.H/2, x.W/2)
        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
        score = self.classifier(score)  # size=(N, n_class, x.H/1, x.W/1)

        return score  # size=(N, n_class, x.H/1, x.W/1)


def main():
    vgg_model = VGGNet()
    model = FCNs(pretrained_net=vgg_model, n_class=2)
    # test the model with random input

    input = torch.randn(1, 3, 224, 224)
    output = model(input)
    print(output.size())


if __name__ == "__main__":
    main()
