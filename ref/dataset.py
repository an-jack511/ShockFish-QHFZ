"""
dataset

generate MN40-classify and/or MN40-retrieval dataset
from OS-MN40-core dataset at the same dir.
"""
from resource import *


def dataset_classify(source_path: Union[str, Path] = "OS-MN40-core/",
                     target_path: Union[str, Path] = "MN40-classify/",
                     k: float = 0.8,
                     clean_target_dir: bool = True) -> None:
    source_path = Path(source_path)
    target_path = Path(target_path)
    if clean_target_dir:
        print("removing existing folder")
        if os.path.exists(target_path):
            shutil.rmtree(target_path)
        os.mkdir(target_path)
    print("generating label info")
    res, label = [], {}
    with open(source_path/"query_label.txt", "r") as f:
        res += str(f.read()).split("\n")
    with open(source_path/"target_label.txt", "r") as f:
        res += str(f.read()).split("\n")
    for lab_root in Path(source_path).glob("train/*"):
        lab = lab_root.name
        for obj in lab_root.glob("*"):
            res.append(obj.name+","+lab)
    for obj in res:
        name, lab = obj.split(",")
        label[name] = lab
    res.sort()
    label_list = sorted(set([obj.split(",")[1] for obj in res]))
    with open(target_path/"all_label.txt", "w") as o:
        for line in res:
            o.write(line+"\n")
    with open(target_path/"label_index.txt", "w") as o:
        for lab in label_list:
            o.write(f"{lab},{label_list.index(lab)}\n")
    del res

    print("copying dataset")
    dir_list, train_list, test_list = [], [], []
    dir_list += list(Path(source_path).glob("query/*"))
    dir_list += list(Path(source_path).glob("target/*"))
    dir_list += list(Path(source_path).glob("train/*/*"))
    cur, tot, t0, lst = 0, len(dir_list), time.time(), 0
    for obj in dir_list:
        train = (random.random() <= k)
        if train:
            if not os.path.exists(target_path/"train"/obj.name):
                shutil.copytree(obj, target_path/"train"/obj.name)
            train_list.append((obj.name, label[obj.name]))
        else:
            if not os.path.exists(target_path/"test"/obj.name):
                shutil.copytree(obj, target_path/"test"/obj.name)
            test_list.append([obj.name, label[obj.name]])
        cur += 1
        if cur-lst >= 100:
            v = 100/(time.time()-t0+0.001)
            print(f"{cur}/{tot} copied | avg speed {v:.2f}/s | time left {(tot-cur)/v:.2f}s")
            print(obj)
            lst, t0 = cur, time.time()
    with open(target_path/"train/label.txt", "w") as o:
        for obj in train_list:
            o.write(f"{obj[0]},{obj[1]}\n")
    with open(target_path/"test/label.txt", "w") as o:
        for obj in test_list:
            o.write(f"{obj[0]},{obj[1]}\n")
    print("finished")


def dataset_retrieval(source_path: Union[str, Path] = "OS-MN40-core",
                      target_path: Union[str, Path] = "MN40-retrieval",
                      k_train: float = 0.8,
                      k_query: float = 0.2,
                      clean_target_dir: bool = True) -> None:
    source_path = Path(source_path)
    target_path = Path(target_path)
    if clean_target_dir:
        print("removing existing folder")
        if os.path.exists(target_path):
            shutil.rmtree(target_path)
        os.mkdir(target_path)
    print("generating label info")
    res, label = [], {}
    with open(source_path/"query_label.txt", "r") as f:
        res += str(f.read()).split("\n")
    with open(source_path/"target_label.txt", "r") as f:
        res += str(f.read()).split("\n")
    for lab_root in Path(source_path).glob("train/*"):
        lab = lab_root.name
        for obj in lab_root.glob("*"):
            res.append(obj.name+","+lab)
    for obj in res:
        name, lab = obj.split(",")
        label[name] = lab
    res.sort()
    label_list = sorted(set([obj.split(",")[1] for obj in res]))
    with open(target_path/"all_label.txt", "w") as o:
        for line in res:
            o.write(line+"\n")
    with open(target_path/"label_index.txt", "w") as o:
        for lab in label_list:
            o.write(f"{lab},{label_list.index(lab)}\n")
    del res

    print("copying dataset")
    dir_list, train_list, query_list, target_list = [], [], [], []
    dir_list += list(Path(source_path).glob("query/*"))
    dir_list += list(Path(source_path).glob("target/*"))
    dir_list += list(Path(source_path).glob("train/*/*"))
    cur, tot, t0, lst = 0, len(dir_list), time.time(), 0
    for obj in dir_list:
        train = (random.random() <= k_train)
        if train:
            if not os.path.exists(target_path/"train"/obj.name):
                shutil.copytree(obj, target_path/"train"/obj.name)
            train_list.append((obj.name, label[obj.name]))
        else:
            query = (random.random() <= k_query)
            if query:
                if not os.path.exists(target_path/"query"/obj.name):
                    shutil.copytree(obj, target_path/"query"/obj.name)
                query_list.append((obj.name, label[obj.name]))
            else:
                if not os.path.exists(target_path/"target"/obj.name):
                    shutil.copytree(obj, target_path/"target"/obj.name)
                target_list.append([obj.name, label[obj.name]])
        cur += 1
        if cur-lst >= 100:
            v = 100/(time.time()-t0+0.001)
            print(f"{cur}/{tot} copied | avg speed {v:.2f}/s | time left {(tot-cur)/v:.2f}s")
            print(obj)
            lst, t0 = cur, time.time()
    with open(target_path/"train/label.txt", "w") as o:
        for obj in train_list:
            o.write(f"{obj[0]},{obj[1]}\n")
    with open(target_path/"query/label.txt", "w") as o:
        for obj in query_list:
            o.write(f"{obj[0]},{obj[1]}\n")
    with open(target_path/"target/label.txt", "w") as o:
        for obj in target_list:
            o.write(f"{obj[0]},{obj[1]}\n")
    print("finished")


if __name__ == "__main__":
    random.seed(2024)
    raise Exception('unexpected launch')
    dataset_retrieval(clean_target_dir=False)
