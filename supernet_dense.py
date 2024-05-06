# image segmentaion with dice + BCE loss deep learning model
# test the model with random input
from torch import nn
from torchvision.models import DenseNet121_Weights, densenet121
from operations import MixedOp
import torch
from param import CONFIG

ranges = {
    "vgg16": ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),
    "densenet": ((0, 3), (4, 6), (6, 8), (8, 10), (10, 12)),
}


class DenseNet(nn.Module):
    def __init__(self, pretrained=True):
        super(DenseNet, self).__init__()
        self.ranges = ranges["densenet"]
        self.features = densenet121(weights=DenseNet121_Weights.DEFAULT).features
        self.conv2d = nn.Conv2d(1024, 1024, kernel_size=1, stride=2, padding=0)

    def forward(self, x):
        output = {}
        for idx in range(len(self.ranges)):
            for layer in range(self.ranges[idx][0], self.ranges[idx][1]):
                x = self.features[layer](x)
            output["x%d" % (idx + 1)] = x
        output["x5"] = self.conv2d(x)
        return output


class SuperNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = DenseNet()
        self.deconv1 = MixedOp(1024, 512)
        self.deconv2 = MixedOp(512, 256)
        self.deconv3 = MixedOp(256, 128)
        self.deconv4 = MixedOp(128, 64)
        self.deconv5 = MixedOp(64, 32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        output = self.pretrained_net(x)

        x5 = output["x5"]  # size=(N, 512, x.H/32, x.W/32)
        x4 = output["x4"]  # size=(N, 512, x.H/16, x.W/16)
        x3 = output["x3"]  # size=(N, 256, x.H/8,  x.W/8)
        x2 = output["x2"]  # size=(N, 128, x.H/4,  x.W/4)
        x1 = output["x1"]  # size=(N, 64, x.H/2,  x.W/2)

        score = self.deconv1(x5)  # size=(N, 512, x.H/16, x.W/16)
        score = score + x4  # element-wise add, size=(N, 512, x.H/16, x.W/16)
        score = self.deconv2(score)  # size=(N, 256, x.H/8, x.W/8)
        score = score + x3  # element-wise add, size=(N, 256, x.H/8, x.W/8)
        score = self.deconv3(score)  # size=(N, 128, x.H/4, x.W/4)
        score = score + x2  # element-wise add, size=(N, 128, x.H/4, x.W/4)
        score = self.deconv4(score)  # size=(N, 64, x.H/2, x.W/2)
        score = score + x1  # element-wise add, size=(N, 64, x.H/2, x.W/2)
        score = self.deconv5(score)  # size=(N, 32, x.H, x.W)
        score = self.classifier(score)  # size=(N, n_class, x.H/1, x.W/1)

        return score  # size=(N, n_class, x.H/1, x.W/1)

    def clip_alphas(self):
        self.deconv1.clip_alphas()
        self.deconv2.clip_alphas()
        self.deconv3.clip_alphas()
        self.deconv4.clip_alphas()
        self.deconv5.clip_alphas()

    def get_alphas(self):
        # return by list
        alpha_list = [
            self.deconv1.alphas.cpu().detach().numpy().tolist(),
            self.deconv2.alphas.cpu().detach().numpy().tolist(),
            self.deconv3.alphas.cpu().detach().numpy().tolist(),
            self.deconv4.alphas.cpu().detach().numpy().tolist(),
            self.deconv5.alphas.cpu().detach().numpy().tolist(),
        ]
        return alpha_list


class SampledNetwork(nn.Module):
    def __init__(self, super_net):
        super(SampledNetwork, self).__init__()

        # VGGNet을 그대로 사용
        if len(CONFIG["GPU"]) >= 2:
            self.pretrained_net = super_net.module.pretrained_net  
        else:
            self.pretrained_net = super_net.pretrained_net

        # SuperNet에서 MixedOp의 선택된 연산을 가져옵니다.
        if len(CONFIG["GPU"]) >= 2:
            self.deconv1 = super_net.module.deconv1.get_max_op()
            self.deconv2 = super_net.module.deconv2.get_max_op()
            self.deconv3 = super_net.module.deconv3.get_max_op()
            self.deconv4 = super_net.module.deconv4.get_max_op()
            self.deconv5 = super_net.module.deconv5.get_max_op()

            self.classifier = super_net.module.classifier
        else:
            self.deconv1 = super_net.deconv1.get_max_op()
            self.deconv2 = super_net.deconv2.get_max_op()
            self.deconv3 = super_net.deconv3.get_max_op()
            self.deconv4 = super_net.deconv4.get_max_op()
            self.deconv5 = super_net.deconv5.get_max_op()

            self.classifier = super_net.classifier

    def forward(self, x):
        output = self.pretrained_net(x)

        x5 = output["x5"]  # size=(N, 512, x.H/32, x.W/32)
        x4 = output["x4"]  # size=(N, 512, x.H/16, x.W/16)
        x3 = output["x3"]  # size=(N, 256, x.H/8,  x.W/8)
        x2 = output["x2"]  # size=(N, 128, x.H/4,  x.W/4)
        x1 = output["x1"]  # size=(N, 64, x.H/2,  x.W/2)

        score = self.deconv1(x5)  # size=(N, 512, x.H/16, x.W/16)
        score = score + x4  # element-wise add, size=(N, 512, x.H/16, x.W/16)
        score = self.deconv2(score)  # size=(N, 256, x.H/8, x.W/8)
        score = score + x3  # element-wise add, size=(N, 256, x.H/8, x.W/8)
        score = self.deconv3(score)  # size=(N, 128, x.H/4, x.W/4)
        score = score + x2  # element-wise add, size=(N, 128, x.H/4, x.W/4)
        score = self.deconv4(score)  # size=(N, 64, x.H/2, x.W/2)
        score = score + x1  # element-wise add, size=(N, 64, x.H/2, x.W/2)
        score = self.deconv5(score)  # size=(N, 32, x.H, x.W)
        score = self.classifier(score)  # size=(N, n_class, x.H/1, x.W/1)

        return score  # size=(N, n_class, x.H/1, x.W/1)


def main():
    vgg_model = VGGNet()
    model = SuperNet(pretrained_net=vgg_model, n_class=2)
    # test the model with random input

    input = torch.randn(1, 3, 224, 224)
    output = model(input)
    print(output.size())


if __name__ == "__main__":
    main()
