#!/usr/bin/env python3
"""
从已下载的 D-Fire 里抽一批真样本, 用来"跑完 Tier A"。

把 N 张 fire/smoke (正) + N 张 None (负/干扰物) 复制到一个文件夹,
并按 fire_/smoke_/none_ 前缀命名, 这样 pilot_query_signal.py 能自动识别标签。
然后正负对比才有统计意义 (之前只有 1 张干扰物)。

注意: 仍是 COCO 权重的探针, 只能给"动机性结论"(看分布是否重叠/分不开),
不能证明 DAQ 成立 —— 那是 Tier B 的事。

用法:
    python scripts/make_tier_a_sample.py --root "D:/path/to/DFire" --n 80 --out pilot_images_dfire
    python scripts/pilot_query_signal.py --weights rtdetr-l.pt --images pilot_images_dfire --out pilot_outputs/query_probe_dfire
"""
import argparse, os, glob, random, shutil, sys

IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


def find_image_dirs(root):
    out = []
    for d, _, _ in os.walk(root):
        if os.path.basename(d).lower() == 'images' and \
           os.path.isdir(os.path.join(os.path.dirname(d), 'labels')):
            out.append(d)
    return out


def label_path_for(img):
    img_dir = os.path.dirname(img)
    lbl_dir = os.path.join(os.path.dirname(img_dir), 'labels')
    stem = os.path.splitext(os.path.basename(img))[0]
    return os.path.join(lbl_dir, stem + '.txt')


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='D-Fire 根目录')
    ap.add_argument('--n', type=int, default=80, help='正/负各抽多少张')
    ap.add_argument('--out', default='pilot_images_dfire')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        sys.exit(f'[error] 找不到目录: {root}')
    img_dirs = find_image_dirs(root)
    if not img_dirs:
        sys.exit(f'[error] {root} 下没找到 images/+labels/ 结构。\n'
                 '把 --root 指到 D-Fire 解压后的上层目录; 若还不行, 把目录树发我。')

    pos_fire, pos_smoke, neg = [], [], []
    for d in img_dirs:
        for ext in IMG_EXT:
            for img in glob.glob(os.path.join(d, '*' + ext)):
                lab = classify(label_path_for(img))
                (pos_fire if lab == 'fire' else pos_smoke if lab == 'smoke' else neg).append(img)

    rng = random.Random(args.seed)
    for lst in (pos_fire, pos_smoke, neg):
        rng.shuffle(lst)
    # 正样本里火/烟各取一半, 凑够 n
    half = args.n // 2
    picks = ([('fire', p) for p in pos_fire[:half]] +
             [('smoke', p) for p in pos_smoke[:args.n - half]] +
             [('none', p) for p in neg[:args.n]])

    os.makedirs(args.out, exist_ok=True)
    cnt = {'fire': 0, 'smoke': 0, 'none': 0}
    for lab, src in picks:
        cnt[lab] += 1
        ext = os.path.splitext(src)[1].lower()
        dst = os.path.join(args.out, f'{lab}_{cnt[lab]:04d}{ext}')
        shutil.copy2(src, dst)

    print('========== Tier A 样本已生成 ==========')
    print(f'来源: {root}')
    print(f'可用总量: fire={len(pos_fire)} smoke={len(pos_smoke)} none={len(neg)}')
    print(f'已复制到 {os.path.abspath(args.out)}: '
          f"fire={cnt['fire']} smoke={cnt['smoke']} none={cnt['none']}")
    if cnt['none'] < args.n or (cnt['fire'] + cnt['smoke']) < args.n:
        print('[警告] 某类样本不足, 已尽量多取; 数量偏少时结论参考性下降。')
    print('\n下一步:')
    print(f'  python scripts/pilot_query_signal.py --weights rtdetr-l.pt '
          f'--images {args.out} --out pilot_outputs/query_probe_dfire')
    print('跑完把 query_probe_dfire/query_separability.json + query_hist.png 发我。')


if __name__ == '__main__':
    main()
