"""
Libs & some other stuff
by: Z.
"""

from __future__ import print_function

import os
import re
import copy
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

import chess

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')