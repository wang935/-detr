#!/usr/bin/env python3
"""
D-Fire 数据准备 (Tier B, baseline-first).

自动识别已下载的 D-Fire 目录结构 (YOLO 格式: 0=smoke, 1=fire), 然后产出:
  - 训练用 Ultralytics 数据集 yaml (两个 arm):
      dfire_full.yaml     全量 train (含 None 干扰物 -> hard-negative arm)
      dfire_posonly.yaml  仅含 fire/smoke 框的图 (naive baseline arm)
  - 误报评测集标签:
      eval_labels.csv     test 划分上的 image,label  (fire|smoke|none)
  - 图像清单 txt: train_full.txt / train_posonly.txt / test.txt

设计要点
--------
* D-Fire 的 None 图 (官方 9,838 张, 含云/夕阳等"像火不是火"场景) 直接作为
  误报评测的负样本, 这正是 DAQ-DETR 干扰物协议要的东西。
* 不复制几 GB 图片: 用 Ultralytics 支持的"图像清单 txt"方式, 训练时按
  images/ -> labels/ 规则自动找标注。所以你的 D-Fire 必须是
  .../images/... 与并列 .../labels/... 结构 (官方/Kaggle 版都是)。
  脚本会自检并在找不到标注时报警。

用法
----
    python scripts/prep_dfire.py --root "D:/path/to/DFire" --out data/dfire
先看它打印的 report, 确认正/负样本数量合理, 再去训练。
"""
import argparse, os, glob, csv, random, sys


IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


def find_image_dirs(root):
    """返回所有名为 images 的目录 (其并列目录应为 labels)。"""
    dirs = []
    for d, subs, _ in os.walk(root):
        if os.path.basename(d).lower() == 'images' and \
           os.path.isdir(os.path.join(os.path.dirname(d), 'labels')):
            dirs.append(d)
    return dirs


def label_path_for(img_path):
    """images/xxx.jpg -> labels/xxx.txt (并列 labels 目录)。"""
    img_dir = os.path.dirname(img_path)
    lbl_dir = os.path.join(os.path.dirname(img_dir), 'labels')
    stem = os.path.splitext(os.path.basename(img_path))[0]
    return os.path.join(lbl_dir, stem + '.txt')


def classify(lbl_path):
    """返回 'fire' / 'smoke' / 'none'  (positive = 标注里有框)。"""
    if not os.path.exists(lbl_path):
        return 'none'
    has_fire = has_smoke = False
    with open(lbl_path) as f:
        for line in f:
            p = line.split()
            if len(p) >= 5:
                cls = p[0]
                if cls == '1':
                    has_fire = True
                elif cls == '0':
                    has_smoke = True
    if has_fire:
        return 'fire'
    if has_smoke:
        return 'smoke'
    return 'none'  # 空标注文件 = None 干扰物


def split_of(path):
    low = path.replace('\\', '/').lower()
    for s in ('train', 'valid', 'val', 'test'):
        if f'/{s}/' in low:
            return 'val' if s in ('valid', 'val') else s
    return 'all'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='已下载的 D-Fire 根目录')
    ap.add_argument('--out', default='data/dfire', help='输出目录')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    if not os.path.isdir(root):
        sys.exit(f'[error] 找不到目录: {root}')

    img_dirs = find_image_dirs(root)
    if not img_dirs:
        sys.exit(f'[error] 在 {root} 下没找到 images/ + labels/ 并列结构。\n'
                 '        D-Fire 官方/Kaggle 版应有 train/images, train/labels 等。\n'
                 '        请确认解压路径, 或把 --root 指到包含它们的上层目录。')

    # 收集 (image, split, label)
    records = []
    for d in img_dirs:
        for ext in IMG_EXT:
            for img in glob.glob(os.path.join(d, '*' + ext)):
                lbl = label_path_for(img)
                records.append((os.path.abspath(img), split_of(img), classify(lbl)))
    if not records:
        sys.exit('[error] images 目录里没有图片。')

    # 标注可发现性自检
    sample = records[:200]
    found = sum(1 for img, _, _ in sample if os.path.exists(label_path_for(img)))
    print(f'[check] 抽样 {len(sample)} 张, 找到标注 {found} 张 '
          f'({"OK" if found else "警告: 0 张找到标注, Ultralytics 训练会失败!"})')

    # 若没有 train/test 划分, 按文件名哈希随机分 (80/20)
    splits = {s for _, s, _ in records}
    if not ({'train', 'test', 'val'} & splits):
        print('[info] 未发现 train/test 子目录, 自动 80/20 划分。')
        rng = random.Random(args.seed)
        records = [(img, ('train' if rng.random() < 0.8 else 'test'), lab)
                   for img, _, lab in records]

    def subset(pred):
        return [img for img, s, lab in records if pred(s, lab)]

    train_full = subset(lambda s, lab: s == 'train')
    train_pos = subset(lambda s, lab: s == 'train' and lab in ('fire', 'smoke'))
    test_imgs = subset(lambda s, lab: s in ('test', 'val'))
    if not test_imgs:  # 只有 train 时, 从 train 切 20% 当 test
        test_imgs = train_full[int(len(train_full) * 0.8):]
        train_full = train_full[:int(len(train_full) * 0.8)]
        train_pos = [i for i in train_pos if i in set(train_full)]

    # 写图像清单
    def write_list(name, items):
        with open(os.path.join(out, name), 'w', encoding='utf-8') as f:
            f.write('\n'.join(items))
        return len(items)

    n_tf = write_list('train_full.txt', train_full)
    n_tp = write_list('train_posonly.txt', train_pos)
    n_te = write_list('test.txt', test_imgs)

    # 写两个 arm 的 yaml
    def write_yaml(name, train_txt):
        with open(os.path.join(out, name), 'w', encoding='utf-8') as f:
            f.write(f"path: {out}\n")
            f.write(f"train: {os.path.join(out, train_txt)}\n")
            f.write(f"val: {os.path.join(out, 'test.txt')}\n")
            f.write("names:\n  0: smoke\n  1: fire\n")
    write_yaml('dfire_full.yaml', 'train_full.txt')
    write_yaml('dfire_posonly.yaml', 'train_posonly.txt')

    # 写评测标签 (test 划分: 正样本 fire/smoke, 负样本 none=干扰物)
    lab_map = {img: lab for img, s, lab in records if s in ('test', 'val')}
    if not lab_map:  # 自切的 test
        lab_map = {img: classify(label_path_for(img)) for img in test_imgs}
    with open(os.path.join(out, 'eval_labels.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['image', 'label'])
        for img in test_imgs:
            w.writerow([img, lab_map.get(img, classify(label_path_for(img)))])

    # 统计报告
    from collections import Counter
    c_all = Counter(lab for _, _, lab in records)
    c_test = Counter(lab_map.get(i, 'none') for i in test_imgs)
    print('\n========== D-Fire 准备报告 ==========')
    print(f'根目录: {root}')
    print(f'总图片: {len(records)}   类别分布: '
          f"fire={c_all['fire']} smoke={c_all['smoke']} none={c_all['none']}")
    print(f'train_full (hard-neg arm): {n_tf} 张 (含 None 干扰物)')
    print(f'train_posonly (baseline arm): {n_tp} 张 (仅 fire/smoke)')
    print(f'test/eval: {n_te} 张  -> 正={c_test["fire"] + c_test["smoke"]} '
          f'负(none/干扰物)={c_test["none"]}')
    print(f'\n输出目录: {out}')
    print('  dfire_full.yaml / dfire_posonly.yaml  -> 两个训练 arm')
    print('  eval_labels.csv                       -> 喂给 fppi_fpr_eval.py')
    print('=====================================')
    if c_test['none'] < 50 or (c_test['fire'] + c_test['smoke']) < 50:
        print('[警告] 评测正或负样本偏少, FPR/FPPI 可能不稳; 检查划分。')
    print('下一步: 看 TIER_B_README.md 第 2 步开始训练两个 arm。')


if __name__ == '__main__':
    main()
