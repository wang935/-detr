#!/usr/bin/env python3
"""
False-alarm evaluation protocol for fire/smoke detection.

The fire-detection literature almost always reports only mAP. For an ALARM system
the decision-relevant question is different: at a recall you are willing to ship
(say 90%), how often does the model fire on a fire-like distractor? This module
reports exactly that:

  - Recall  : fraction of POSITIVE (fire/smoke) images with >=1 detection >= thr
  - FPR     : fraction of NEGATIVE (distractor/none) images with >=1 detection >= thr
  - FPPI    : mean number of false detections per negative image
  - and the headline: FPR / FPPI measured AT a FIXED image-level recall (.80/.85/.90/.95)

For paper-grade reporting, choose the recall threshold on a calibration label
file and report FPR/FPPI on a disjoint test label file. Passing only --labels
keeps the historical in-sample behavior for quick diagnostics and should not be
used for formal claims.

Image-level recall is the right early-warning metric; for box-level mAP use
`yolo val` / RT-DETR's own validator. This protocol is what differentiates the
paper, so keep it model-agnostic: feed it any model's predictions.

INPUT
-----
  --pred  predictions.csv   columns: image,conf   (one row per detection;
                            images with no detection may be omitted)
  --labels labels.csv       columns: image,label  (label in fire|smoke|distractor|none)
  --calib-labels labels.csv optional calibration labels for threshold choice
  --test-labels labels.csv  optional disjoint test labels for reporting

USAGE
-----
  python scripts/fppi_fpr_eval.py --pred baseline_preds.csv --labels labels.csv \
    --calib-labels eval_calib_labels.csv --test-labels eval_test_labels.csv --out report.json
  # head-to-head (this is the go/no-go comparison):
  python scripts/fppi_fpr_eval.py --pred hardneg_preds.csv --pred2 daq_preds.csv --labels labels.csv \
    --calib-labels eval_calib_labels.csv --test-labels eval_test_labels.csv
"""
import argparse, csv, json
from collections import defaultdict
import numpy as np

MIN_THRESHOLD = 1e-6

def load(pred_csv, labels_csv):
    lab = {}
    for r in csv.DictReader(open(labels_csv)):
        lab[r['image'].strip()] = r['label'].strip().lower()
    dets = defaultdict(list)
    for r in csv.DictReader(open(pred_csv)):
        dets[r['image'].strip()].append(float(r['conf']))
    return lab, dets


def _max(dets, img):
    v = dets.get(img, [])
    return max(v) if v else 0.0


def metrics_at(thr, pos_imgs, neg_imgs, dets):
    recall = np.mean([1.0 if _max(dets, i) >= thr else 0.0 for i in pos_imgs]) if pos_imgs else 0.0
    fpr = np.mean([1.0 if _max(dets, i) >= thr else 0.0 for i in neg_imgs]) if neg_imgs else 0.0
    fppi = np.mean([sum(1 for c in dets.get(i, []) if c >= thr) for i in neg_imgs]) if neg_imgs else 0.0
    return float(recall), float(fpr), float(fppi)


def evaluate(lab, dets, targets=(0.80, 0.85, 0.90, 0.95), test_lab=None):
    test_lab = test_lab or lab
    pos_imgs = [i for i, l in lab.items() if l in ('fire', 'smoke')]
    test_pos_imgs = [i for i, l in test_lab.items() if l in ('fire', 'smoke')]
    test_neg_imgs = [i for i, l in test_lab.items() if l in ('distractor', 'none')]
    # candidate thresholds: every observed positive max-conf + a dense grid
    observed = {_max(dets, i) for i in pos_imgs if _max(dets, i) > MIN_THRESHOLD}
    grid = {float(x) for x in np.linspace(0, 1, 201) if x > MIN_THRESHOLD}
    posmax = sorted(observed | grid, reverse=True)
    out = {
        'n_pos': len(test_pos_imgs),
        'n_neg': len(test_neg_imgs),
        'n_calib_pos': len(pos_imgs),
        'eval_protocol': 'legacy_in_sample' if test_lab is lab else 'calibration_threshold_test_report',
        'at_fixed_recall': {},
    }
    for tr in targets:
        chosen = None
        # thresholds descending => recall non-decreasing. Take the FIRST (highest) thr
        # that already meets the recall target => minimal false alarms at that recall.
        for thr in posmax:
            calib_rc = metrics_at(thr, pos_imgs, [], dets)[0]
            if calib_rc >= tr:
                rc, fpr, fppi = metrics_at(thr, test_pos_imgs, test_neg_imgs, dets)
                chosen = {'thr': round(float(thr), 4), 'calib_recall': round(calib_rc, 4),
                          'recall': round(rc, 4),
                          'FPR': round(fpr, 4), 'FPPI': round(fppi, 4)}
                break
        out['at_fixed_recall'][f'{tr:.2f}'] = chosen
    return out


def print_report(name, rep):
    print(f'\n=== {name}  (pos={rep["n_pos"]}, neg={rep["n_neg"]}) ===')
    print(f'{"target recall":>14} | {"thr":>6} | {"recall":>6} | {"FPR":>6} | {"FPPI":>6}')
    for tr, m in rep['at_fixed_recall'].items():
        if m:
            print(f'{tr:>14} | {m["thr"]:>6} | {m["recall"]:>6} | {m["FPR"]:>6} | {m["FPPI"]:>6}')
        else:
            print(f'{tr:>14} |  (recall target not reachable)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True, help='predictions CSV: image,conf')
    ap.add_argument('--pred2', default=None, help='second model for head-to-head')
    ap.add_argument('--labels', required=True, help='labels CSV: image,label')
    ap.add_argument('--calib-labels', default=None, help='calibration labels for threshold choice')
    ap.add_argument('--test-labels', default=None, help='test labels for FPR/FPPI reporting')
    ap.add_argument('--out', default=None, help='optional JSON output path')
    args = ap.parse_args()
    if bool(args.calib_labels) != bool(args.test_labels):
        raise SystemExit('[error] --calib-labels and --test-labels must be provided together for split evaluation')

    lab, dets = load(args.pred, args.calib_labels or args.labels)
    test_lab = None
    if args.test_labels:
        test_lab, _ = load(args.pred, args.test_labels)
    rep = evaluate(lab, dets, test_lab=test_lab)
    print_report(args.pred, rep)
    result = {'model_A': {'pred': args.pred, **rep}}

    if args.pred2:
        _, dets2 = load(args.pred2, args.calib_labels or args.labels)
        rep2 = evaluate(lab, dets2, test_lab=test_lab)
        print_report(args.pred2, rep2)
        result['model_B'] = {'pred': args.pred2, **rep2}
        # verdict at recall 0.90
        a = rep['at_fixed_recall'].get('0.90')
        b = rep2['at_fixed_recall'].get('0.90')
        if a and b:
            d = b['FPR'] - a['FPR']
            print(f'\n[verdict @recall0.90] {args.pred} FPR={a["FPR"]}  vs  '
                  f'{args.pred2} FPR={b["FPR"]}  (delta={d:+.4f})')
            print('If pred2 is DAQ and pred is hard-negative+threshold and delta<0 by a clear '
                  'margin, DAQ earns its place. If delta~0, DAQ is NOT needed -> kill (per plan).')

    if args.out:
        json.dump(result, open(args.out, 'w'), indent=2)
        print(f'\n[done] wrote {args.out}')


if __name__ == '__main__':
    main()
