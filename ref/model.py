"""
model

build the neural network structure
"""
from resource import *


class Multiview(nn.Module):
    def __init__(self, drop_rate: float):
        super(Multiview, self).__init__()
        # input shape = (batch_size, views, vox_len, vox_len) = (40, 8, 224, 224)
        self.base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.feat = nn.Sequential(*list(self.base.children())[:-1])
        self.fc = nn.Linear(modality_config['multiview']['views']*512, 1200)

        self.cls = nn.Linear(1200, 40)

        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=drop_rate)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = x.view(-1, 3, modality_config['multiview']['img_len'], modality_config['multiview']['img_len'])
        x = self.drop(self.feat(x))

        x = x.view(-1, modality_config['multiview']['views']*512)
        feat = self.drop(self.relu(self.fc(x)))

        x = self.drop(self.relu(self.cls(feat)))
        return x, feat


class PointCloud(nn.Module):
    def __init__(self, drop_rate: float):
        super(PointCloud, self).__init__()
        # input shape = (batch_size, dimension, pts) = (40, 3, 1024)
        self.conv1 = nn.Conv1d(3, 72, 1)
        self.conv2 = nn.Conv1d(72, 360, 1)
        self.conv3 = nn.Conv1d(360, 1200, 1)
        # self.fc = nn.Linear(1200, 1200)

        self.cls = nn.Linear(1200, 40)

        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=drop_rate)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.drop(self.relu(self.conv1(x)))
        x = self.drop(self.relu(self.conv2(x)))
        x = self.drop(self.relu(self.conv3(x)))
        feat = torch.max(x, 2, keepdim=True)[0].view(-1, 1200)
        # x = self.drop(self.relu(self.fc(x)))

        x = self.drop(self.relu(self.cls(feat)))
        return x, feat


class Voxel(nn.Module):
    def __init__(self, drop_rate: float):
        super(Voxel, self).__init__()
        # input shape = (batch_size, 1, vox_len, vox_len, vox_len) = (40, 1, 32, 32, 32)
        self.conv1 = nn.Conv3d(1, 4, 3)
        self.conv2 = nn.Conv3d(4, 8, 3)
        self.fc = nn.Linear(1728, 1200)

        self.cls = nn.Linear(1200, 40)

        self.relu = nn.ReLU()
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.drop(func.max_pool3d(self.relu(self.conv1(x)), 2))
        x = self.drop(func.max_pool3d(self.relu(self.conv2(x)), 2))
        x = torch.flatten(x, start_dim=1)
        feat = self.drop(self.relu(self.fc(x)))

        x = self.drop(self.relu(self.cls(feat)))
        return x, feat


class ObjectRetrieval(nn.Module):
    def __init__(self, drop_rate: float = 0.5):
        t0 = time.time()
        super(ObjectRetrieval, self).__init__()
        self.multiview = Multiview(drop_rate)
        self.point_cloud = PointCloud(drop_rate)
        self.voxel = Voxel(drop_rate)
        self.modality = {
            "multiview": self.multiview,
            "point_cloud": self.point_cloud,
            "voxel": self.voxel
        }

        self.fc1 = nn.Linear(3600, 2400)
        self.fc2 = nn.Linear(2400, 1200)
        self.fc3 = nn.Linear(1200, 40)

        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=drop_rate)
        print(f"new model ready in {time.time()-t0:.2f}s\n"
              f"dropout rate = {drop_rate:.2f}")

    def forward(self, x: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, Tensor]]:
        feat = {}
        for key in x.keys():
            _, feat[key] = self.modality[key](x[key])
        x = torch.cat(list(feat.values()), 1)
        x = self.drop(self.relu(self.fc1(x)))
        x = self.drop(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x, feat


if __name__ == "__main__":
    net = ObjectRetrieval().to(device)
    print(net)
    test = {'multiview': torch.rand((batch_size, modality_config['multiview']['views'], 3, modality_config['multiview']['img_len'], modality_config['multiview']['img_len']), dtype=torch.float32, device=device),
            'point_cloud': torch.rand((batch_size, 3, modality_config['point_cloud']['pts']), dtype=torch.float32, device=device),
            'voxel': torch.rand((batch_size, 1, modality_config['voxel']['vox_len'], modality_config['voxel']['vox_len'], modality_config['voxel']['vox_len']), dtype=torch.float32, device=device)}
    out = net(test)
