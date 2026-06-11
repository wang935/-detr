#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 DAQ-DETR  Tier A  --  单文件全流程 (含 RTX 3060 环境安装)
================================================================================
把这一个文件发到 3060 主机即可。它包含:
  1) 环境安装 (--install): 装 CUDA 版 torch (适配 3060/Ampere) + ultralytics 等
  2) 从 D-Fire 抽样 (--dfire-root): 抽 N 张 fire/smoke + N 张 None(干扰物)
  3) Tier A 探针: 抽取 RT-DETR 的 300 个 object-query 置信度分布,
     比较 火/烟 vs 干扰物 的可分性, 出 CSV / JSON / 直方图

⚠ 诚实说明: rtdetr-l.pt 是 COCO 权重 (没有 fire/smoke 类)。本探针衡量的是
  "像不像 COCO 物体", 不是"像不像火"。所以 Tier A 只能给【动机性结论】——
  预期会看到火和干扰物的 query 分布【重叠、分不开】, 正好证明"现成 DETR 不够,
  必须微调 + DAQ"。它【不能】证明 DAQ 成立, 那是 Tier B (D-Fire 微调) 的事。

--------------------------------------------------------------------------------
 用法 (Windows / Linux 通用)
--------------------------------------------------------------------------------
  # 第一次: 装环境 (建议先建虚拟环境/conda 环境再装)
  python tier_a_3060.py --install

  # 方式 A: 已经有 D-Fire -> 自动抽样并跑 (推荐)
  python tier_a_3060.py --dfire-root "D:\\path\\to\\DFire" --n 80

  # 方式 B: 自己准备好图片文件夹 (文件名以 fire_/smoke_/distractor_/none_ 开头)
  python tier_a_3060.py --images my_images

  rtdetr-l.pt 会由 ultralytics 自动下载, 无需手动拷贝 (主机需联网)。
  跑完把  out 目录下的 query_separability.json + query_hist.png 发回来判读。
================================================================================
"""
import argparse
import os
import sys
import glob

IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


# ------------------------------------------------------------------ 环境安装 --
def do_install():
    """装 CUDA 版 torch (cu121, 适配 RTX 3060/Ampere) + ultralytics 等。"""
    import subprocess
    py = sys.executable
    print(f'[install] 使用解释器: {py}')
    steps = [
        [py, '-m', 'pip', 'install', '--upgrade', 'pip'],
        # cu121 wheel 同时提供 Windows / Linux 版, pip 自动选; 3060 完全支持
        [py, '-m', 'pip', 'install', 'torch', 'torchvision',
         '--index-url', 'https://download.pytorch.org/whl/cu121'],
        [py, '-m', 'pip', 'install', 'ultralytics', 'matplotlib', 'numpy'],
    ]
    for cmd in steps:
        print('[install] $', ' '.join(cmd))
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print('[install] 失败。若是 Windows 且想用 CPU, 可改用默认源: '
                  'pip install torch torchvision; 但 Tier A 建议用 GPU。')
            sys.exit(r.returncode)
    # 验证
    try:
        import torch
        print(f'[install] OK  torch={torch.__version__}  '
              f'cuda_available={torch.cuda.is_available()}  '
              f'gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}')
    except Exception as e:
        print(f'[install] 装完但导入 torch 失败: {e}')
    print('[install] 完成。现在可以跑: '
          'python tier_a_3060.py --dfire-root "D:\\path\\to\\DFire" --n 80')


# ------------------------------------------------------------- D-Fire 抽样 --
def find_image_dirs(root):
    out = []
    for d, _, _ in os.walk(root):
        if os.path.basename(d).lower() == 'images' and \
           os.path.isdir(os.path.join(os.path.dirname(d), 'labels')):
            out.append(d)
    return out


def label_path_for(img):
    return os.path.join(os.path.dirname(os.path.dirname(img)), 'labels',
                        os.path.splitext(os.path.basename(img))[0] + '.txt')


def classify(lbl):
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


def sample_from_dfire(root, n, out, seed=0):
    import random
    import shutil
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        sys.exit(f'[error] 找不到 D-Fire 目录: {root}')
    dirs = find_image_dirs(root)
    if not dirs:
        sys.exit(f'[error] {root} 下没找到 images/+labels/ 结构。\n'
                 '        把 --dfire-root 指到 D-Fire 解压后的上层目录;\n'
                 '        若还不行, 把该文件夹的目录树发我, 我改匹配规则。')
    fire, smoke, none = [], [], []
    for d in dirs:
        for ext in IMG_EXT:
            for img in glob.glob(os.path.join(d, '*' + ext)):
                lab = classify(label_path_for(img))
                (fire if lab == 'fire' else smoke if lab == 'smoke' else none).append(img)
    rng = random.Random(seed)
    for lst in (fire, smoke, none):
        rng.shuffle(lst)
    half = n // 2
    picks = ([('fire', p) for p in fire[:half]] +
             [('smoke', p) for p in smoke[:n - half]] +
             [('none', p) for p in none[:n]])
    os.makedirs(out, exist_ok=True)
    cnt = {'fire': 0, 'smoke': 0, 'none': 0}
    for lab, src in picks:
        cnt[lab] += 1
        ext = os.path.splitext(src)[1].lower()
        shutil.copy2(src, os.path.join(out, f'{lab}_{cnt[lab]:04d}{ext}'))
    print(f'[sample] 来源 {root}  可用: fire={len(fire)} smoke={len(smoke)} none={len(none)}')
    print(f'[sample] 复制到 {os.path.abspath(out)}: '
          f"fire={cnt['fire']} smoke={cnt['smoke']} none={cnt['none']}")
    if cnt['none'] < n or cnt['fire'] + cnt['smoke'] < n:
        print('[sample][警告] 某类不足, 已尽量取; 数量少则结论参考性下降。')
    return out


# --------------------------------------------------------- Tier A 探针 --
def label_from_name(name):
    base = os.path.basename(name).lower()
    for L in ('distractor', 'smoke', 'fire', 'none'):
        if base.startswith(L):
            return L
    return 'unknown'


def _entropy(p):
    import numpy as np
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def qfeatures(conf):
    import numpy as np
    conf = np.sort(np.asarray(conf, dtype=float))[::-1]
    if conf.size == 0:
        return dict(n_q=0, n_25=0, n_50=0, top1=0.0, top5=0.0, top20=0.0,
                    mean=0.0, std=0.0, tail_lt10=0.0, ent=0.0)
    topk = lambda k: float(conf[:k].mean())
    hist, _ = np.histogram(conf, bins=20, range=(0, 1))
    p = hist / max(hist.sum(), 1)
    return dict(
        n_q=int(conf.size), n_25=int((conf >= 0.25).sum()), n_50=int((conf >= 0.50).sum()),
        top1=float(conf[0]), top5=topk(min(5, conf.size)), top20=topk(min(20, conf.size)),
        mean=float(conf.mean()), std=float(conf.std()),
        tail_lt10=float((conf < 0.10).mean()), ent=_entropy(p),
    )


def run_probe(weights, images, out, imgsz=640):
    import json
    import csv
    import statistics as st
    import numpy as np
    from ultralytics import RTDETR
    import torch

    os.makedirs(out, exist_ok=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    gpu = torch.cuda.get_device_name(0) if dev == 'cuda' else 'CPU'
    print(f'[probe] device={dev} ({gpu})  weights={weights}')
    if dev == 'cpu':
        print('[probe][警告] 没检测到 CUDA, 在 CPU 上跑会很慢。先 --install 装 GPU 版 torch。')
    model = RTDETR(weights)

    paths = []
    for ext in IMG_EXT:
        paths += glob.glob(os.path.join(images, '*' + ext))
    paths = sorted(paths)
    if not paths:
        sys.exit(f'[error] {images} 里没有图片。')
    print(f'[probe] {len(paths)} 张图片')

    rows, raw = [], {}
    for p in paths:
        # conf=0.001 + max_det=300: RT-DETR 无 NMS, 据此基本拿到全部 300 个 query 的分数
        r = model.predict(p, conf=0.001, max_det=300, imgsz=imgsz, device=dev, verbose=False)[0]
        conf = r.boxes.conf.detach().cpu().numpy() if r.boxes is not None else np.array([])
        lab = label_from_name(p)
        feat = qfeatures(conf)
        feat['image'] = os.path.basename(p)
        feat['label'] = lab
        rows.append(feat)
        raw[feat['image']] = np.sort(np.asarray(conf, dtype=float))[::-1]

    np.savez(os.path.join(out, 'query_vectors.npz'),
             **{k.replace('.', '_'): v for k, v in raw.items()})

    keys = ['image', 'label', 'n_q', 'n_25', 'n_50', 'top1', 'top5', 'top20',
            'mean', 'std', 'tail_lt10', 'ent']
    with open(os.path.join(out, 'query_features.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in keys})

    pos = [r for r in rows if r['label'] in ('fire', 'smoke')]
    neg = [r for r in rows if r['label'] in ('distractor', 'none')]
    summary = {'weights': weights, 'n_pos': len(pos), 'n_neg': len(neg), 'features': {}}
    print(f'\n[separability] 正(火/烟 n={len(pos)}) vs 负(干扰物/none n={len(neg)}) -- |gap| 越大信号越强')
    if pos and neg:
        for k in ('top1', 'top5', 'top20', 'mean', 'n_25', 'tail_lt10', 'ent'):
            mp = st.mean(r[k] for r in pos)
            mn = st.mean(r[k] for r in neg)
            summary['features'][k] = {'pos_mean': mp, 'neg_mean': mn, 'gap': mp - mn}
            print(f'  {k:<10} pos={mp:.3f}  neg={mn:.3f}  gap={mp - mn:+.3f}')
    else:
        print('  [警告] 缺正或负样本; 用 --dfire-root 抽样, 或确认文件名前缀。')
    json.dump(summary, open(os.path.join(out, 'query_separability.json'), 'w'), indent=2)

    # 直方图: 正 vs 负 的平均 query 分布
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        bins = np.linspace(0, 1, 21)

        def avg_hist(group):
            hs = [np.histogram(raw[r['image']], bins=bins, density=True)[0]
                  for r in group if raw[r['image']].size]
            return np.mean(hs, axis=0) if hs else np.zeros(len(bins) - 1)

        if pos and neg:
            c = (bins[:-1] + bins[1:]) / 2
            plt.figure(figsize=(7, 4))
            plt.plot(c, avg_hist(pos), '-o', label=f'fire/smoke (n={len(pos)})')
            plt.plot(c, avg_hist(neg), '-s', label=f'distractor/none (n={len(neg)})')
            plt.xlabel('per-query confidence')
            plt.ylabel('avg density')
            plt.title('RT-DETR query-score distribution: positive vs negative')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(out, 'query_hist.png'), dpi=130)
            print(f'[probe] 写出 {os.path.join(out, "query_hist.png")}')
    except Exception as e:
        print(f'[probe] 直方图跳过 ({e}); CSV/JSON/npz 已保存。')

    if pos and neg:
        gaps = {k: abs(v['gap']) for k, v in summary['features'].items()}
        best = max(gaps, key=gaps.get)
        print(f'\n[takeaway] 最强区分特征 = "{best}" (gap={summary["features"][best]["gap"]:+.3f})。'
              ' COCO 权重下这只证明"信号可抽取、有结构", 不是真信号。')
    print(f'[done] 输出在 {os.path.abspath(out)}: '
          'query_features.csv / query_separability.json / query_vectors.npz / query_hist.png')
    print('把 query_separability.json + query_hist.png 发回来判读, 给 Tier A 收尾。')


# ----------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser(description='DAQ-DETR Tier A 单文件全流程 (含 3060 环境安装)')
    ap.add_argument('--install', action='store_true', help='安装 CUDA 版 torch + ultralytics 等')
    ap.add_argument('--dfire-root', default=None, help='D-Fire 根目录; 给了就自动抽样')
    ap.add_argument('--n', type=int, default=80, help='正/负各抽多少张 (配合 --dfire-root)')
    ap.add_argument('--images', default='pilot_images_dfire', help='探针输入图片文件夹')
    ap.add_argument('--weights', default='rtdetr-l.pt', help='权重 (默认自动下载 COCO RT-DETR-L)')
    ap.add_argument('--out', default='pilot_outputs/query_probe_dfire', help='输出目录')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    if args.install:
        do_install()
        return
    if args.dfire_root:
        sample_from_dfire(args.dfire_root, args.n, args.images, args.seed)
    run_probe(args.weights, args.images, args.out, args.imgsz)


if __name__ == '__main__':
    main()
