"""
DNN + MCTS
by: Z.
"""

from headers.utils import *


class ResNetBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, int, int] = (3, 3, 3)):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size)

        self.bn1 = nn.BatchNorm3d(out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.skip_connection = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        residual = self.skip_connection(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return self.relu(x)


class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.conv = nn.Sequential(*[
            nn.Conv3d(1, 256, (3, 3, 3)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True)
        ])
        self.resnet_blocks = nn.Sequential(*[ResNetBlock(256, 256) for _ in range(19)])

        self.policy_head = nn.Sequential(*[
            nn.Conv3d(256, 2, (1, 1, 1)),
            nn.BatchNorm3d(2),
            nn.Tanh()
        ])
        self.value_head = nn.Sequential(*[
            nn.Conv3d(256, 2, (1, 1, 1)),
            nn.BatchNorm3d(2),
            nn.Softmax()
        ])

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.conv(x)
        x = self.resnet_blocks(x)
        policy, value = self.policy_head(x), self.value_head(x)
        return policy, value


if __name__ == "__main__":
    net = DNN()
    print(net)
