"""
train retrieval

train & eval. the modal for the retrieval task
"""
from resource import *

from ref.model import ObjectRetrieval
from ref.dataloader import MN40_retrieval
from ref.loss_function import CombinedLoss
root_path = "./MN40-retrieval"


def setup(print_log: bool = True) -> None:
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    if print_log:
        print(f"random seed = {random_seed}")


def train(data_loader: DataLoader,
          net: nn.Module,
          criterion: nn.Module,
          optimizer: Union[optim.Optimizer, Dict[str, optim.Optimizer]],
          drop: Union[None, str] = None) -> Tuple[float, float]:
    t0 = time.time()
    net.train()
    tot_loss = 0
    t1 = t0
    for i, (lbl, mv, pc, vx) in enumerate(data_loader):
        lbl, mv, pc, vx = lbl.to(device), mv.to(device), pc.to(device), vx.to(device)
        data = {
            "multiview": mv,
            "point_cloud": pc,
            "voxel": vx
        }
        if drop in data.keys():
            data[drop] = torch.zeros_like(data[drop])

        optimizer.zero_grad()
        res, feat = net(data)
        loss, c_loss, h_loss = criterion(res, feat, lbl)
        loss.backward()
        optimizer.step()

        loss = loss.item()
        print("\r"
              f"batch {i+1:d}/{len(data_loader):d}  "
              f"time = {time.time()-t1:.2f}s  "
              f"loss = {loss:.4f} ({c_loss:.4f}, {h_loss:.4f})", end='')
        tot_loss += loss
        t1 = time.time()
    tot_time = time.time()-t0
    avg_loss = tot_loss/len(data_loader)
    print("\r"
          f"time = {tot_time:.2f}s (batch_avg = {tot_time/len(data_loader):.2f}s)\n"
          f"drop = {drop}\n"
          f"loss = {avg_loss:.4f}\n")

    return tot_time, avg_loss


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
         drop: Union[None, str] = None) -> Tuple[float, float, float]:
    t0 = time.time()
    print("feature extraction (1/2)")
    q_ft, q_lbl = extract_feature(query_loader, net, drop)
    print("feature extraction (2/2)")
    t_ft, t_lbl = extract_feature(target_loader, net, drop)
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
    top1 = top_k_acc(q_lbl, t_lbl, idx, k=1)
    top5 = top_k_acc(q_lbl, t_lbl, idx, k=5)
    print(f"mAP score = {mAP:.4f} ({time.time()-t1:.2f}s)\n"
          f"top 1 accuracy = {top1*100:.2f}% ({time.time()-t1:.2f}s)\n"
          f"top 5 accuracy = {top5*100:.2f}% ({time.time()-t1:.2f}s)\n"
          f"evaluation complete ({time.time()-t0:.2f}s)")
    return mAP, top1, top5


def track_display(path: Union[Path, str],
                  show_gui: bool = True,
                  output_file: bool = True) -> None:
    path = Path(path)
    data = pd.read_csv(path/'track.csv', dtype={'checkpoint #': int, 'file': str, 'loss ratio': str, 'mAP score': float, 'top1 accuracy': float, 'top5 accuracy': float})
    data = data.to_numpy()

    plt.xlim(0, data.shape[0])
    plt.xlabel("checkpoint", loc="center")
    plt.ylim(0, 1)
    plt.plot(data[:, 3], label="mAP")
    plt.plot(data[:, 4], label="top1 acc")
    plt.plot(data[:, 5], label="top5 acc")
    plt.legend()
    plt.title("performance track")
    if output_file:
        plt.savefig(path/"track.png")
        print(f"figure saved at \"{path}\\track.png\"")
    if show_gui:
        plt.show()


def main() -> None:
    setup()
    print("load dataset")
    train_loader, query_loader, target_loader = MN40_retrieval(modality=modality_config, batch_size=batch_size)
    # '''
    print("create new model")
    net = ObjectRetrieval(dropout_rate)
    save_path = create_save_path(mode='retrieval')
    chkpt, tot_chkpt = 0, max_epoch//chkpt_interval
    '''
    while True:
        try:
            s = input("restore checkpoint: ")
            net = torch.load(s)
            break
        except FileNotFoundError:
            print("invalid directory")
    save_path = Path(os.path.dirname(s))
    chkpt = int(re.findall(r'\d+', s)[-1])
    tot_chkpt = chkpt+max_epoch//chkpt_interval
    # '''

    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.2)
    criterion = CombinedLoss()
    net.to(device)
    criterion.to(device)

    with open(save_path/'config.txt', 'w') as cfg:
        cfg.write('==========config==========\n')
        cfg.write('hyper-params:')
        cfg.write(config_info())
        cfg.write('\n==========network==========\n')
        print(net, file=cfg)
        cfg.write('\n==========optimizer==========\n')
        print(optimizer, file=cfg)
        cfg.write('\n==========criterion==========\n')
        print(criterion, file=cfg)

    with open(save_path/'track.csv', "w") as log:
        log.write("checkpoint #,file,loss ratio,mAP score,top1 accuracy,top5 accuracy")

    for epoch in range(1, max_epoch+1):
        torch.cuda.empty_cache()
        drop = random.choice(modality_drop)
        print(f"drop = {drop}\n"
              f"train epoch {epoch}/{max_epoch}...")
        train(train_loader, net, criterion, optimizer, drop=drop)

        if epoch % chkpt_interval == 0:
            chkpt += 1
            chkpt_path = save_path/f"chkpt-{chkpt}.nn"
            torch.save(net, chkpt_path)
            print(f"checkpoint #{chkpt} saved at \"{chkpt_path}\"\n"
                  f"evaluate checkpoint {chkpt:d}/{max_epoch//chkpt_interval:d}")
            mAP, top1, top5 = test(query_loader, target_loader, net, drop=drop)
            with open(save_path/'track.csv', "a") as log:
                log.write(f"\n{chkpt},{chkpt_path},{criterion.ratio:.4f}:{1-criterion.ratio:.4f},{mAP:.4f},{top1:.4f},{top5:.4f}")
            criterion.update_ratio()
            print('track log updated')

    print("finished")
    track_display(save_path, show_gui=False)


if __name__ == "__main__":
    main()
