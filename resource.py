"""
resource

all libs, packs and utils are imported and defined here
"""
from __future__ import print_function

import os
import re
import time
import json
import scipy
import random
import shutil
import numpy as np
import pandas as pd
import open3d as o3d
from PIL import Image

from torch import Tensor
from pathlib import Path
from matplotlib import pyplot as plt
from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as func
from torch.utils.data import Dataset, DataLoader

import torchvision.models as models
import torchvision.transforms as tf

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

full_modality = {
    'multiview': {'views': 24,
                  'img_len': 256},
    'point_cloud': {'pts': 1024},
    'voxel': {'vox_len': 32}
}
modality_config = {
    'multiview': {'views': 8,
                  'img_len': 224},
    'point_cloud': {'pts': 1024},
    'voxel': {'vox_len': 32}
}
random_seed = 2024
learning_rate = 2e-3
dropout_rate = 0.25

loss_ratio = 1.0
delta_ratio = 0.05
min_ratio = 0.75

batch_size = 16
max_epoch = 100
chkpt_interval = 5

modality_drop = ['multiview']*0+['voxel']*0+['point_cloud']*0
modality_drop = modality_drop+[None]*(100-len(modality_drop))


def create_save_path(mode: str) -> Path:
    if mode not in ['retrieval']:
        raise RuntimeError('invalid mode')
    path = Path(time.strftime(f"./models/{mode}/%Y%m%d-%H%M%S"))
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def config_info() -> str:
    config = {
        'random seed': random_seed,
        'learning rate': learning_rate,
        'dropout rate': dropout_rate,
        'loss ratio': f"{loss_ratio} -> {min_ratio}, delta= {delta_ratio}",
        'batch size': batch_size,
        'max epoch': max_epoch,
        'checkpoint interval': chkpt_interval,
        'modality': modality_config,
        'modality drop rate': {
            'multiview': round(modality_drop.count('multiview')/len(modality_drop), 2),
            'point cloud': round(modality_drop.count('point_cloud')/len(modality_drop), 2),
            'voxel': round(modality_drop.count('voxel')/len(modality_drop), 2)
        }
    }
    return json.dumps(config, indent=2, ensure_ascii=False)[2:-1]


if __name__ == "__main__":
    print("config info:", config_info(), sep='\n')
