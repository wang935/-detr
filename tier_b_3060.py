#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 DAQ-DETR  Tier B  --  单文件全流程 (含 RTX 3060 环境安装)
================================================================================
Tier B = 真正的 go / no-go。在 D-Fire 上微调 RT-DETR, 用「固定召回下的
误报率 FPR / FPPI」比较两个 arm, 判断 DAQ 校准头到底有没有必要做。

本阶段只做 baseline 与 hard-negative 两个 arm (用户已确认"先基线、验 kill-gate")。
DAQ 校准头是 Stage 2, 只有当 hard-neg 仍留下明显误报 gap 时才上 —— 见末尾说明。

⚠ kill-gate (你自己定的): 若「hard-negative 训练 + 调阈值」就把误报降到和
  baseline 差不多低, 则 DAQ 没必要做 -> 本方向应转 FPQ-DETR。

--------------------------------------------------------------------------------
 子命令 (按顺序跑)
--------------------------------------------------------------------------------
  python tier_b_3060.py install
      装 CUDA 版 torch (cu121, 适配 3060) + ultralytics + numpy

  python tier_b_3060.py prep --dfire-root "D:\\fire\\D-Fire" --out data\\dfire
      自动识别 D-Fire(YOLO: 0=smoke,1=fire), 产出:
        dfire_posonly.yaml  baseline arm (仅 fire/smoke 图)
        dfire_full.yaml     hard-neg arm (含 None 干扰物图)
        eval_labels.csv     test 划分的 image,label (喂给 eval)

  python tier_b_3060.py train --arm baseline --data data\\dfire\\dfire_posonly.yaml --epochs 60 --batch 8
  python tier_b_3060.py train --arm hardneg  --data data\\dfire\\dfire_full.yaml    --epochs 60 --batch 8
      微调 RT-DETR。两个 arm 必须同模型同超参, 只差训练集 -> 对比才公平。
      显存: 3060-12G 用 batch 8; 3060-6G(笔记本) 用 batch 4, imgsz 也可降到 512。

  python tier_b_3060.py export --weights runs_tierb\\baseline\\weights\\best.pt --labels data\\dfire\\eval_labels.csv --out preds\\baseline.csv
  python tier_b_3060.py export --weights runs_tierb\\hardneg\\weights\\best.pt  --labels data\\dfire\\eval_labels.csv --out preds\\hardneg.csv
      在 test 集上跑推理, 导出 image,conf (任一类的框=报警)。

  python tier_b_3060.py eval --pred preds\\baseline.csv --pred2 preds\\hardneg.csv --labels data\\dfire\\eval_labels.csv --out gonogo.json
      算固定召回(0.80/0.85/0.90/0.95)下的 FPR / FPPI, 并给 go/no-go 结论。
================================================================================
"""
import argparse
import os
import sys
import glob
import csv
import math

IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


# ============================================================ 通用 / 环境 ==
def _is_bad_python():
    e = sys.executable.lower().replace('\\', '/')
    return any(t in e for t in ('/msys', 'mingw', 'ucrt64', 'msys64'))


_BAD_PY = (
    '[error] 当前 Python 是 MSYS2/MinGW 版 (%s),\n'
    '        没有 pip 也跑不了 PyTorch GPU。请改用 conda 或 python.org 版:\n'
    '          conda create -n fire python=3.10 -y && conda activate fire\n'
    '          cd /d D:\\detr_Q3 && python tier_b_3060.py install\n'
)


def _need(mods):
    """缺依赖时给可操作的中文提示, 而不是裸 traceback。"""
    missing = []
    for m in mods:
        try:
            __import__(m)
        except Exception:
            missing.append(m)
    if missing:
        sys.exit(f'[error] 缺少依赖: {missing}。先跑:  python tier_b_3060.py install')


def cmd_install(args):
    import subprocess
    py = sys.executable
    print(f'[install] 解释器: {py}')
    if _is_bad_python():
        sys.exit(_BAD_PY % py)
    try:
        import pip  # noqa
    except ModuleNotFoundError:
        subprocess.run([py, '-m', 'ensurepip', '--upgrade'])
    steps = [
        [py, '-m', 'pip', 'install', '--upgrade', 'pip'],
        [py, '-m', 'pip', 'install', 'torch', 'torchvision',
         '--index-url', 'https://download.pytorch.org/whl/cu121'],
        [py, '-m', 'pip', 'install', 'ultralytics', 'numpy'],
    ]
    for c in steps:
        print('[install] $', ' '.join(c))
        if subprocess.run(c).returncode != 0:
            sys.exit('[install] 失败, 见上方报错。')
    try:
        import torch
        print(f'[install] OK torch={torch.__version__} cuda={torch.cuda.is_available()} '
              f'gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}')
    except Exception as e:
        print(f'[install] 装完导入 torch 失败: {e}')


# ================================================================= prep ==
def _find_image_dirs(root):
    out = []
    for d, _, _ in os.walk(root):
        if os.path.basename(d).lower() == 'images' and \
           os.path.isdir(os.path.join(os.path.dirname(d), 'labels')):
            out.append(d)
    return out


def _label_path_for(img):
    return os.path.join(os.path.dirname(os.path.dirname(img)), 'labels',
                        os.path.splitext(os.path.basename(img))[0] + '.txt')


def _classify(lbl):
    if not os.path.exists(lbl):
        return 'none'
    fire = smoke = False
    with open(lbl) as f:
        for line in f:
            p = line.split()
            if len(p) >= 5:
                if p[0] == '1':
                    fire = True
                elif p[0] == '0':
                    smoke = True
    return 'fire' if fire else ('smoke' if smoke else 'none')


def _valid_yolo_label(lbl):
    """Reject malformed labels that can make detector losses become NaN."""
    if not os.path.exists(lbl) or os.path.getsize(lbl) == 0:
        return True
    try:
        with open(lbl) as f:
            for line in f:
                p = line.split()
                if not p:
                    continue
                if len(p) != 5:
                    return False
                vals = [float(x) for x in p]
                cls, x, y, w, h = vals
                if cls not in (0.0, 1.0):
                    return False
                if not all(math.isfinite(v) for v in vals):
                    return False
                if not all(0.0 <= v <= 1.0 for v in (x, y, w, h)):
                    return False
                if w <= 0.0 or h <= 0.0:
                    return False
    except Exception:
        return False
    return True


def _split_of(path):
    low = path.replace('\\', '/').lower()
    for s in ('train', 'valid', 'val', 'test'):
        if f'/{s}/' in low:
            return 'val' if s in ('valid', 'val') else s
    return 'all'


def cmd_prep(args):
    import random
    from collections import Counter
    root = os.path.abspath(args.dfire_root)
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    if not os.path.isdir(root):
        sys.exit(f'[error] 找不到 D-Fire 目录: {root}')
    dirs = _find_image_dirs(root)
    if not dirs:
        sys.exit(f'[error] {root} 下没有 images/+labels/ 结构。把目录树发我再调。')

    records = []
    bad_records = []
    for d in dirs:
        for ext in IMG_EXT:
            for img in glob.glob(os.path.join(d, '*' + ext)):
                img = os.path.abspath(img)
                lbl = _label_path_for(img)
                if not _valid_yolo_label(lbl):
                    bad_records.append(img)
                    continue
                records.append((img, _split_of(img), _classify(lbl)))
    if not records:
        sys.exit('[error] images 目录里没有图片。')
    if bad_records:
        print(f'[check] 跳过 {len(bad_records)} 张非法 YOLO 标注图 '
              '(坐标越界/零面积/格式错误), 避免训练 loss=nan。')

    sample = records[:200]
    found = sum(1 for img, _, _ in sample if os.path.exists(_label_path_for(img)))
    print(f'[check] 抽样 {len(sample)} 张, 找到标注 {found} 张 '
          f'({"OK" if found else "警告: 0 张找到标注, 训练会失败!"})')

    splits = {s for _, s, _ in records}
    if not ({'train', 'test', 'val'} & splits):
        print('[info] 没有 train/test 子目录, 自动 80/20 划分。')
        rng = random.Random(args.seed)
        records = [(img, ('train' if rng.random() < 0.8 else 'test'), lab)
                   for img, _, lab in records]

    sub = lambda pred: [img for img, s, lab in records if pred(s, lab)]
    train_full = sub(lambda s, lab: s == 'train')
    train_pos = sub(lambda s, lab: s == 'train' and lab in ('fire', 'smoke'))
    test_imgs = sub(lambda s, lab: s in ('test', 'val'))
    if not test_imgs:
        cut = int(len(train_full) * 0.8)
        test_imgs, train_full = train_full[cut:], train_full[:cut]
        train_pos = [i for i in train_pos if i in set(train_full)]

    def wl(name, items):
        with open(os.path.join(out, name), 'w', encoding='utf-8') as f:
            f.write('\n'.join(items))
        return len(items)
    n_tf = wl('train_full.txt', train_full)
    n_tp = wl('train_posonly.txt', train_pos)
    n_te = wl('test.txt', test_imgs)

    def wy(name, train_txt):
        with open(os.path.join(out, name), 'w', encoding='utf-8') as f:
            f.write(f"path: {out}\n")
            f.write(f"train: {os.path.join(out, train_txt)}\n")
            f.write(f"val: {os.path.join(out, 'test.txt')}\n")
            f.write("names:\n  0: smoke\n  1: fire\n")
    wy('dfire_full.yaml', 'train_full.txt')
    wy('dfire_posonly.yaml', 'train_posonly.txt')

    lab_map = {img: lab for img, s, lab in records if s in ('test', 'val')} or \
              {img: _classify(_label_path_for(img)) for img in test_imgs}
    with open(os.path.join(out, 'eval_labels.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['image', 'label'])
        for img in test_imgs:
            w.writerow([img, lab_map.get(img, _classify(_label_path_for(img)))])

    c_all = Counter(lab for _, _, lab in records)
    c_test = Counter(lab_map.get(i, 'none') for i in test_imgs)
    print('\n========== D-Fire 准备报告 ==========')
    print(f'根目录: {root}')
    print(f"总图片: {len(records)}  fire={c_all['fire']} smoke={c_all['smoke']} none={c_all['none']}")
    print(f'train_full (hard-neg arm): {n_tf} 张 (含 None 干扰物)')
    print(f'train_posonly (baseline arm): {n_tp} 张 (仅 fire/smoke)')
    print(f"test/eval: {n_te} 张 -> 正={c_test['fire'] + c_test['smoke']} 负(none)={c_test['none']}")
    print(f'输出: {out}  (dfire_*.yaml, eval_labels.csv)')
    if c_test['none'] < 50 or c_test['fire'] + c_test['smoke'] < 50:
        print('[警告] 评测正/负样本偏少, FPR/FPPI 可能不稳。')
    print('下一步: train --arm baseline ...  然后  train --arm hardneg ...')


# ================================================================ train ==
def cmd_train(args):
    _need(('torch', 'ultralytics'))
    import torch
    from ultralytics import RTDETR
    if not os.path.exists(args.data):
        sys.exit(f'[error] 找不到 data yaml: {args.data} (先跑 prep)')
    dev = 0 if torch.cuda.is_available() else 'cpu'
    gpu = torch.cuda.get_device_name(0) if dev == 0 else 'CPU'
    print(f'[train] arm={args.arm} device={dev}({gpu}) model={args.weights} '
          f'epochs={args.epochs} batch={args.batch} imgsz={args.imgsz}')
    if dev == 'cpu':
        print('[train][警告] 没 CUDA, 训练会非常慢。先 install。')
    model = RTDETR(args.weights)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, device=dev, name=args.arm,
                project=args.project, exist_ok=True, amp=True,
                mosaic=args.mosaic, val=args.val, workers=args.workers,
                plots=args.plots)
    print(f'[train] 完成。权重: {os.path.join(args.project, args.arm, "weights", "best.pt")}')


# =============================================================== export ==
def cmd_export(args):
    _need(('torch', 'ultralytics'))
    import torch
    from ultralytics import RTDETR
    if not os.path.exists(args.weights):
        sys.exit(f'[error] 找不到权重: {args.weights}')
    imgs = [r['image'] for r in csv.DictReader(open(args.labels, encoding='utf-8'))]
    if args.limit:
        imgs = imgs[:args.limit]
    dev = 0 if torch.cuda.is_available() else 'cpu'
    print(f'[export] {len(imgs)} 张 eval 图, device={dev}, conf>={args.conf}, max_det={args.max_det}')
    model = RTDETR(args.weights)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    n_rows = 0
    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['image', 'conf'])
        for i, img in enumerate(imgs):
            r = model.predict(img, conf=args.conf, max_det=args.max_det,
                              imgsz=args.imgsz, device=dev, verbose=False)[0]
            if r.boxes is not None and len(r.boxes):
                for c in r.boxes.conf.detach().cpu().tolist():
                    w.writerow([img, c])
                    n_rows += 1
            if (i + 1) % 200 == 0:
                print(f'  ...{i + 1}/{len(imgs)}')
    print(f'[export] 写出 {args.out} ({n_rows} 条检测; 任一类的框都算报警)')


# ================================================================= eval ==
def _load(pred_csv, labels_csv):
    from collections import defaultdict
    lab = {r['image'].strip(): r['label'].strip().lower()
           for r in csv.DictReader(open(labels_csv, encoding='utf-8'))}
    dets = defaultdict(list)
    for r in csv.DictReader(open(pred_csv, encoding='utf-8')):
        dets[r['image'].strip()].append(float(r['conf']))
    return lab, dets


def _maxc(dets, img):
    v = dets.get(img, [])
    return max(v) if v else 0.0


def _eval_one(lab, dets, targets=(0.80, 0.85, 0.90, 0.95)):
    import numpy as np
    pos = [i for i, l in lab.items() if l in ('fire', 'smoke')]
    neg = [i for i, l in lab.items() if l in ('distractor', 'none')]
    thrs = sorted({_maxc(dets, i) for i in pos} | set(np.linspace(0, 1, 201)), reverse=True)
    out = {'n_pos': len(pos), 'n_neg': len(neg), 'at_fixed_recall': {}}
    for tr in targets:
        chosen = None
        for thr in thrs:
            rc = np.mean([1.0 if _maxc(dets, i) >= thr else 0.0 for i in pos]) if pos else 0.0
            if rc >= tr:
                fpr = np.mean([1.0 if _maxc(dets, i) >= thr else 0.0 for i in neg]) if neg else 0.0
                fppi = np.mean([sum(1 for c in dets.get(i, []) if c >= thr) for i in neg]) if neg else 0.0
                chosen = {'thr': round(float(thr), 4), 'recall': round(float(rc), 4),
                          'FPR': round(float(fpr), 4), 'FPPI': round(float(fppi), 4)}
                break
        out['at_fixed_recall'][f'{tr:.2f}'] = chosen
    return out


def _print_rep(name, rep):
    print(f'\n=== {name}  (pos={rep["n_pos"]}, neg={rep["n_neg"]}) ===')
    print(f'{"目标recall":>10} | {"thr":>6} | {"recall":>6} | {"FPR":>6} | {"FPPI":>6}')
    for tr, m in rep['at_fixed_recall'].items():
        if m:
            print(f'{tr:>10} | {m["thr"]:>6} | {m["recall"]:>6} | {m["FPR"]:>6} | {m["FPPI"]:>6}')
        else:
            print(f'{tr:>10} |  (该召回达不到)')


def cmd_eval(args):
    _need(('numpy',))
    import json
    lab, dets = _load(args.pred, args.labels)
    rep = _eval_one(lab, dets)
    _print_rep(args.pred, rep)
    result = {'model_A': {'pred': args.pred, **rep}}
    if args.pred2:
        _, dets2 = _load(args.pred2, args.labels)
        rep2 = _eval_one(lab, dets2)
        _print_rep(args.pred2, rep2)
        result['model_B'] = {'pred': args.pred2, **rep2}
        a = rep['at_fixed_recall'].get('0.90')
        b = rep2['at_fixed_recall'].get('0.90')
        if a and b:
            d = b['FPR'] - a['FPR']
            print(f'\n[go/no-go @recall0.90] A({args.pred}) FPR={a["FPR"]}  '
                  f'vs  B({args.pred2}) FPR={b["FPR"]}  delta={d:+.4f}')
            print('  约定: A=baseline(posonly), B=hardneg。')
            print('  若 hardneg 把 FPR 压到和后续 DAQ 差不多低 -> DAQ 没必要 -> kill, 转 FPQ。')
            print('  若 hardneg 仍留明显 FPR gap -> 有空间, 进 Stage 2 加 DAQ 校准头。')
    if args.out:
        json.dump(result, open(args.out, 'w'), indent=2)
        print(f'[eval] 写出 {args.out}')


# ================================================================= main ==
def main():
    ap = argparse.ArgumentParser(description='DAQ-DETR Tier B 单文件全流程 (3060)')
    sub = ap.add_subparsers(dest='cmd', required=True)

    sub.add_parser('install', help='装 CUDA torch + ultralytics + numpy')

    p = sub.add_parser('prep', help='识别 D-Fire 并产出 yaml + eval_labels.csv')
    p.add_argument('--dfire-root', required=True)
    p.add_argument('--out', default='data/dfire')
    p.add_argument('--seed', type=int, default=0)

    p = sub.add_parser('train', help='微调 RT-DETR 一个 arm')
    p.add_argument('--arm', required=True, choices=['baseline', 'hardneg'])
    p.add_argument('--data', required=True)
    p.add_argument('--weights', default='rtdetr-l.pt')
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--project', default='runs_tierb')
    p.add_argument('--mosaic', type=float, default=1.0)
    p.add_argument('--val', type=lambda x: str(x).lower() in ('1', 'true', 'yes', 'y'), default=True)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--plots', type=lambda x: str(x).lower() in ('1', 'true', 'yes', 'y'), default=True)

    p = sub.add_parser('export', help='在 eval 集导出 image,conf')
    p.add_argument('--weights', required=True)
    p.add_argument('--labels', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--conf', type=float, default=0.05)
    p.add_argument('--max-det', type=int, default=100)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--limit', type=int, default=0, help='只跑前 N 张 (冒烟测试用)')

    p = sub.add_parser('eval', help='算固定召回下的 FPR/FPPI 并给 go/no-go')
    p.add_argument('--pred', required=True)
    p.add_argument('--pred2', default=None)
    p.add_argument('--labels', required=True)
    p.add_argument('--out', default=None)

    args = ap.parse_args()
    {'install': cmd_install, 'prep': cmd_prep, 'train': cmd_train,
     'export': cmd_export, 'eval': cmd_eval}[args.cmd](args)


if __name__ == '__main__':
    main()
