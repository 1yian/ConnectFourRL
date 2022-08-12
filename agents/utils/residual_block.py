import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, filters=256, kernel_size=3, activation=nn.functional.leaky_relu):
        super(ResidualBlock, self).__init__()
        self.kernel_size = kernel_size
        self.filters = filters
        self.padding = (kernel_size - 1) // 2

        self.activation = activation

        self.conv1 = nn.Conv2d(self.filters, self.filters, self.kernel_size, 1, padding=self.padding)
        self.conv1_bn = nn.BatchNorm2d(self.filters)

        self.conv2 = nn.Conv2d(self.filters, self.filters, self.kernel_size, 1, padding=self.padding)
        self.conv2_bn = nn.BatchNorm2d(self.filters)

    def forward(self, x):
        init_x = x
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = x + init_x
        x = self.activation(x)
        return x
