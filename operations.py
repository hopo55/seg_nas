# darts - mixed operations
# search space - 3 filters (3, 5, 7)

import torch
import torch.nn as nn


class MixedOp(nn.Module):
    def __init__(self, C_in, C_out, stride=2, dilation=1, output_padding=1, bias=False):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        self._ops.append(
            nn.ConvTranspose2d(
                C_in,
                C_out,
                3,
                stride=stride,
                padding=1,
                dilation=dilation,
                output_padding=output_padding,
                bias=bias,
            )
        )
        self._ops.append(
            nn.ConvTranspose2d(
                C_in,
                C_out,
                5,
                stride=stride,
                padding=2,
                dilation=dilation,
                output_padding=output_padding,
                bias=bias,
            )
        )
        self._ops.append(
            nn.ConvTranspose2d(
                C_in,
                C_out,
                7,
                stride=stride,
                padding=3,
                dilation=dilation,
                output_padding=output_padding,
                bias=bias,
            )
        )

        self.bn = nn.BatchNorm2d(C_out)
        self.relu = nn.ReLU(inplace=True)
        self.alphas = nn.Parameter(
            torch.Tensor([1.0 / 3, 1.0 / 3, 1.0 / 3]).cuda(), requires_grad=True
        )

    def clip_alphas(self):
        with torch.no_grad():
            self.alphas.clamp_(0, 1)
            alpha_sum = self.alphas.sum()
            self.alphas.div_(alpha_sum)

    def forward(self, x):
        x = sum(alpha * op(x) for alpha, op in zip(self.alphas, self._ops))
        x = self.relu(x)
        x = self.bn(x)
        return x
    
    def get_max_alpha_idx(self):
        # return the index of the maximum alpha
        return torch.argmax(self.alphas).item()

    def get_max_op(self):
        # return the operation with the maximum alpha
        return self._ops[self.get_max_alpha_idx()]
