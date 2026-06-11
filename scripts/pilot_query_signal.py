#!/usr/bin/env python3
"""
DAQ-DETR Tier-A pilot: RT-DETR query-level confidence-DISTRIBUTION probe.

WHY THIS EXISTS
---------------
DAQ-DETR's thesis is that a DETR detector carries signal in the *distribution*
of its 300 object-query scores (not only in the single top detection), and that
this distribution can separate a true fire/smoke scene from a fire-like
DISTRACTOR scene (sunset, headlight, red object, reflection) better than a plain
confidence threshold. This script extracts that per-query distribution and
measures how separable the two scene types are.

HONESTY NOTE (read before trusting the numbers)
-----------------------------------------------
rtdetr-l.pt is a COCO model with NO fire/smoke class. On COCO weights this
script validates the *instrumentation* -- it proves the query distribution is
extractable and shows structure -- and gives a first look. It is NOT the real
go/no-go. The real test needs an RT-DETR fine-tuned on D-Fire; see
PILOT_QUERY_README.md (Tier B). Run this first to confirm the pipeline, then
re-run it with your D-Fire weights via --weights.

RUN (local, GPU)
----------------
    python scripts/pilot_query_signal.py \
        --weights rtdetr-l.pt --images pilot_images --out pilot_outputs/query_probe

Labels are inferred from filename prefix: fire_ / smoke_ / distractor_ / none_.
Otherwise pass --labels labels.csv  (columns: image,label).
"""
import argparse, os, glob, json, csv
import numpy as np


def entropy(p):
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def qfeatures(conf):
    """Distributional features of one image's per-query confidence vector."""
    conf = np.sort(np.asarray(conf, dtype=float))[::-1]  # descending
    if conf.size == 0:
        return dict(n_q=0, n_25=0, n_50=0, top1=0.0, top5=0.0, top20=0.0,
                    mean=0.0, std=0.0, tail_lt10=0.0, ent=0.0)
    topk = lambda k: float(conf[:k].mean())
    hist, _ = np.histogram(conf, bins=20, range=(0, 1))
    p = hist / max(hist.sum(), 1)
    return dict(
        n_q=int(conf.size),
        n_25=int((conf >= 0.25).sum()),          # # queries that look like real objects
        n_50=int((conf >= 0.50).sum()),
        top1=float(conf[0]),                      # strongest query
        top5=topk(min(5, conf.size)),
        top20=topk(min(20, conf.size)),
        mean=float(conf.mean()),
        std=float(conf.std()),
        tail_lt10=float((conf < 0.10).mean()),    # fraction of "unmatched-like" low queries
        ent=entropy(p),                            # spread of the score distribution
    )


def label_from_name(name):
    base = os.path.basename(name).lower()
    for L in ('distractor', 'smoke', 'fire', 'none'):
        if base.startswith(L):
            return L
    return 'unknown'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default='rtdetr-l.pt')
    ap.add_argument('--images', default='pilot_images')
    ap.add_argument('--out', default='pilot_outputs/query_probe')
    ap.add_argument('--labels', default=None, help='optional CSV: image,label')
    ap.add_argument('--imgsz', type=int, default=640)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from ultralytics import RTDETR
    import torch
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[info] device={dev}  weights={args.weights}')
    model = RTDETR(args.weights)

    labels = {}
    if args.labels and os.path.exists(args.labels):
        for row in csv.DictReader(open(args.labels)):
            labels[os.path.basename(row['image'])] = row['label'].strip().lower()

    paths = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
        paths += glob.glob(os.path.join(args.images, ext))
    paths = sorted(paths)
    print(f'[info] {len(paths)} images in {args.images}')

    rows = []
    raw = {}  # image -> sorted (desc) per-query confidence vector, for plots / paper figures
    for p in paths:
        # conf=0.001 + max_det=300 recovers (almost) the FULL per-query score vector,
        # because RT-DETR is NMS-free and emits a fixed set of object queries.
        r = model.predict(p, conf=0.001, max_det=300, imgsz=args.imgsz,
                          device=dev, verbose=False)[0]
        conf = r.boxes.conf.detach().cpu().numpy() if r.boxes is not None else np.array([])
        lab = labels.get(os.path.basename(p), label_from_name(p))
        feat = qfeatures(conf)
        feat['image'] = os.path.basename(p)
        feat['label'] = lab
        rows.append(feat)
        raw[feat['image']] = np.sort(np.asarray(conf, dtype=float))[::-1]
        print(f"  {feat['image']:<24} {lab:<11} n_q={feat['n_q']:>3} "
              f"n@.25={feat['n_25']:>3} top5={feat['top5']:.3f} tail<.1={feat['tail_lt10']:.2f}")

    # save raw per-query vectors (paper figures / further analysis)
    np.savez(os.path.join(args.out, 'query_vectors.npz'),
             **{k.replace('.', '_'): v for k, v in raw.items()})

    keys = ['image', 'label', 'n_q', 'n_25', 'n_50', 'top1', 'top5', 'top20',
            'mean', 'std', 'tail_lt10', 'ent']
    with open(os.path.join(args.out, 'query_features.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in keys})

    # Separability: positives (fire/smoke) vs negatives (distractor/none) on each feature.
    import statistics as st
    pos = [r for r in rows if r['label'] in ('fire', 'smoke')]
    neg = [r for r in rows if r['label'] in ('distractor', 'none')]
    summary = {'weights': args.weights, 'n_pos': len(pos), 'n_neg': len(neg), 'features': {}}
    if pos and neg:
        print('\n[separability] positive(fire/smoke) vs negative(distractor/none) '
              '-- larger |gap| = more usable query signal')
        for k in ('top1', 'top5', 'top20', 'mean', 'n_25', 'tail_lt10', 'ent'):
            mp = st.mean(r[k] for r in pos)
            mn = st.mean(r[k] for r in neg)
            summary['features'][k] = {'pos_mean': mp, 'neg_mean': mn, 'gap': mp - mn}
            print(f'  {k:<10} pos={mp:.3f}  neg={mn:.3f}  gap={mp - mn:+.3f}')
    else:
        print('\n[separability] need both positive and negative images to compare; '
              'pass --labels or name files fire_/smoke_/distractor_/none_.')
    json.dump(summary, open(os.path.join(args.out, 'query_separability.json'), 'w'), indent=2)

    # optional histogram: averaged query-score distribution, positives vs negatives
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        bins = np.linspace(0, 1, 21)

        def avg_hist(group):
            hs = []
            for r in group:
                v = raw[r['image']]
                if v.size:
                    h, _ = np.histogram(v, bins=bins, density=True)
                    hs.append(h)
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
            plt.savefig(os.path.join(args.out, 'query_hist.png'), dpi=130)
            print(f'[info] wrote {args.out}/query_hist.png')
    except Exception as e:
        print(f'[info] histogram skipped ({e}); features/vectors still saved.')

    # one-line takeaway based on the strongest separating feature
    if pos and neg:
        gaps = {k: abs(v['gap']) for k, v in summary['features'].items()}
        best = max(gaps, key=gaps.get)
        g = summary['features'][best]['gap']
        print(f'\n[takeaway] strongest separating feature = "{best}" (gap={g:+.3f}). '
              'On COCO weights this only proves the signal is extractable & structured.')
    print(f'[done] wrote {args.out}/query_features.csv, query_separability.json, query_vectors.npz')
    print('Next: re-run with your D-Fire-fine-tuned weights (--weights runs/.../best.pt) '
          'for the REAL signal; a clear positive/negative gap there is the DAQ go-signal.')


if __name__ == '__main__':
    main()
