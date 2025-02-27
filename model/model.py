"""
DNN + MCTS
by: Z.
"""

from utils import *


class ResBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, int, int] = (3, 3, 3)):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size)

        self.bn1 = nn.BatchNorm3d(out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.skip_connection = nn.Identity()

    def forward(self, x: Tensor):
        residual = self.skip_connection(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return self.relu(x)


class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        conv1 = nn.Conv3d(1, 256, (3, 3, 3))
        bn = nn.BatchNorm3d(256)
        relu = nn.ReLU()


