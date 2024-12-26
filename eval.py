"""
eval

evaluate trained model checkpoints from cache
and generate result chart
"""
from resource import *

from model import ObjectRetrieval
from dataloader import MN40_retrieval
root_path = "./MN40-retrieval"


@torch.no_grad()
def extract_feature(data_loader: DataLoader,
                    net: nn.Module,
                    drop: Union[None, str],
                    print_log: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    t0 = time.time()
    net.eval()
    all_lbl, all_ft = [], []
    t1 = time.time()
    for i, (lbl, mv, pc, vx) in enumerate(data_loader):
        mv, pc, vx = mv.cuda(), pc.cuda(), vx.cuda()
        data = {
            "multiview": mv,
            "point_cloud": pc,
            "voxel": vx
        }
        if drop in data.keys():
            data[drop] = torch.zeros_like(data[drop])
        _, feat = net(data)
        feat = torch.cat(list(feat.values()), dim=1)
        all_ft.append(feat.detach().cpu().numpy())
        all_lbl.append(lbl.numpy())
        if print_log:
            print("\r"
                  f"batch {i+1:2d}/{len(data_loader):2d}  "
                  f"time = {time.time()-t1:.2f}s", end='')
        t1 = time.time()
    all_lbl = np.concatenate(all_lbl, axis=0)
    all_ft = np.concatenate(all_ft, axis=0)
    if print_log:
        print(f"\rdone ({time.time()-t0:.2f})s{' '*50}")
    return all_ft, all_lbl


def top_k_acc(x_lbl: np.ndarray,
              y_lbl: np.ndarray,
              idx: np.ndarray,
              k: int = 1) -> float:
    x, y = x_lbl.shape[0], y_lbl.shape[0]
    k = min(k, y)
    acc = np.zeros(x)
    for i in range(x):
        cur_idx = idx[i]
        cnt = 0
        for j in range(k):
            cnt += (x_lbl[i] == y_lbl[cur_idx[j]])
        acc[i] = (cnt/k)
    acc = np.average(acc)
    return acc


def test(query_loader: DataLoader,
         target_loader: DataLoader,
         net: nn.Module,
         drop: Union[None, str] = None,
         print_log: bool = False) -> Tuple[float, float, float]:
    t0 = time.time()
    if print_log:
        print("feature extraction (1/2)")
    q_ft, q_lbl = extract_feature(query_loader, net, drop, print_log=print_log)
    if print_log:
        print("feature extraction (2/2)")
    t_ft, t_lbl = extract_feature(target_loader, net, drop, print_log=print_log)
    if print_log:
        print("start evaluation")
    t1 = time.time()
    dis_mat = scipy.spatial.distance.cdist(q_ft, t_ft, metric='euclidean')
    x, y = dis_mat.shape
    idx = dis_mat.argsort()
    ap = []
    for i in range(x):
        cur_idx = idx[i]
        p = []
        r = 0
        for j in range(y):
            if q_lbl[i] == t_lbl[cur_idx[j]]:
                r += 1
                p.append(r/(j+1))
        if r > 0:
            for j in range(len(p)):
                p[j] = max(p[j:])
            ap.append(np.array(p).mean())
    mAP = np.mean(ap)
    t1 = time.time()
    top1 = top_k_acc(q_lbl, t_lbl, idx, k=1)
    t1 = time.time()
    top5 = top_k_acc(q_lbl, t_lbl, idx, k=5)
    if print_log:
        print(f"top 1 accuracy = {top1*100:.2f}% ({time.time()-t1:.2f}s)\n"
              f"mAP score = {mAP:.4f} ({time.time()-t1:.2f}s)\n"
              f"top 5 accuracy = {top5*100:.2f}% ({time.time()-t1:.2f}s)\n"
              f"evaluation complete ({time.time()-t0:.2f}s)")
    return mAP, top1, top5


def main():
    print("load dataset")
    _, query_loader, target_loader = MN40_retrieval(modality=modality_config, batch_size=batch_size)
    print("load model")
    while True:
        try:
            s = input("evaluate model: ")
            net = torch.load(s)
            break
        except FileNotFoundError:
            print("invalid directory")
    net.to(device)
    print("evaluating...")
    for drop in [None, 'multiview', 'point cloud', 'voxel']:
        mAP, top1, top5 = test(query_loader, target_loader, net, drop=drop)
        print(f"drop = {drop} | mAP = {mAP:.4f} | top1 acc = {top1:.4f} | top5 acc = {top5:.4f}")


if __name__ == "__main__":
    main()
