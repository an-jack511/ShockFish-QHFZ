"""
dataloader

load the MN40-classify and/or MN40-retrieval dataset from file
and generate PyTorch DataLoader for train and eval.
"""
from resource import *


class Loader(Dataset):
    def __init__(self,
                 root: Union[str, Path],
                 modality: Dict[str, Dict],
                 set_type: str):
        super().__init__()
        root = Path(root)
        self.root = root/set_type
        self.modality = modality
        self.label_idx = {}
        with open(root/"label_index.txt") as f:
            for line in f.readlines():
                lab, idx = line.strip().split(",")
                self.label_idx[lab] = int(idx)
        self.objects = []
        with open(self.root/"label.txt") as f:
            for line in f.readlines():
                obj, lab = line.strip().split(",")
                self.objects.append({
                    "path": obj,
                    "label": lab,
                    "label_idx": self.label_idx[lab]
                })

    def load_img(self, path: Union[str, Path]) -> Tensor:
        path = self.root/path/"image"
        filename = list(path.glob("h_*.jpg"))
        tot, views, img_len = len(filename), self.modality["multiview"]["views"], self.modality["multiview"]["img_len"]
        filename = filename[::tot//views][:views]
        transform = tf.Compose([tf.Resize(img_len), tf.ToTensor(), ])
        img = [transform(Image.open(i).convert("RGB")) for i in filename]
        return torch.stack(img).view(-1, img_len, img_len)

    def load_point_cloud(self, path: Union[str, Path]) -> Tensor:
        path = self.root/path/"pointcloud"
        pts = self.modality["point_cloud"]["pts"]
        filename = path/f"pt_{pts}.pts"
        pt = np.asarray(o3d.io.read_point_cloud(str(filename)).points)
        pt -= np.expand_dims(np.mean(pt, axis=0), 0)
        pt /= np.max(np.sqrt(np.sum(pt**2, axis=1)), 0)
        return torch.from_numpy(pt.astype(np.float32)).transpose(0, 1)

    def load_voxel(self, path: Union[str, Path]) -> Tensor:
        path = self.root/path/"voxel"
        vox_len = self.modality["voxel"]["vox_len"]
        filename = path/f"vox_{vox_len}.ply"
        vox_3d = o3d.io.read_voxel_grid(str(filename))
        idx = torch.from_numpy(np.array([i.grid_index-1 for i in vox_3d.get_voxels()])).long()
        vox = torch.zeros((vox_len, vox_len, vox_len))
        vox[idx[:, 0], idx[:, 1], idx[:, 2]] = 1
        return vox.view(1, 32, 32, 32)

    def __getitem__(self, idx: int) -> List[Tensor]:
        obj = self.objects[idx]
        path, data = obj["path"], [obj["label_idx"]]
        for key in self.modality.keys():
            if key == "multiview":
                img = self.load_img(path)
                data.append(img)
            elif key == "point_cloud":
                pc = self.load_point_cloud(path)
                data.append(pc)
            elif key == "voxel":
                vx = self.load_voxel(path)
                data.append(vx)
        return data

    def __len__(self):
        return len(self.objects)


def MN40_classify(modality: Dict[str, Dict[str, int]],
                  root: Union[str, Path] = "MN40-legacy",
                  batch_size: int = 32,
                  print_log: bool = False) -> Tuple[DataLoader, DataLoader]:
    t0 = time.time()
    train_set = Loader(root, modality, set_type="train")
    test_set = Loader(root, modality, set_type="test")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=2)
    if print_log:
        print(f"train samples: {len(train_set)}")
        print(f"test samples: {len(test_set)}")
        print(f"MN40-legacy dataset loaded in {time.time()-t0:.2f}s")
    return train_loader, test_loader


def MN40_retrieval(modality: Dict[str, Dict[str, int]],
                   root: Union[str, Path] = "MN40-retrieval",
                   batch_size: int = 32,
                   print_log: bool = False) -> Tuple[DataLoader, DataLoader, DataLoader]:
    t0 = time.time()
    train_set = Loader(root, modality, set_type="train")
    query_set = Loader(root, modality, set_type="query")
    target_set = Loader(root, modality, set_type="target")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)
    query_loader = DataLoader(query_set, batch_size=batch_size, shuffle=False, num_workers=2)
    target_loader = DataLoader(target_set, batch_size=batch_size, shuffle=False, num_workers=4)
    if print_log:
        print(f"train samples: {len(train_set)}")
        print(f"query samples: {len(query_set)}")
        print(f"target samples: {len(target_set)}")
        print(f"MN40-retrieval dataset loaded in {time.time()-t0:.2f}s")
    return train_loader, query_loader, target_loader


if __name__ == "__main__":
    print(MN40_classify(modality=full_modality, print_log=True))
    print(MN40_retrieval(modality=full_modality, print_log=True))
