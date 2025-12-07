from typing import Dict, List, Tuple


Box = Tuple[int, int, int, int]


def iou(b1: Box, b2: Box) -> float:
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0

    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


def load_annotations_txt(path: str) -> Dict[int, List[Tuple[str, Box]]]:
    ann = {}
    with open(path, "r") as f:
        for line in f:
            p = line.strip().split()
            if len(p) != 6:
                continue
            fid = int(p[0])
            cname = p[1]
            box = tuple(map(int, p[2:]))
            ann.setdefault(fid, []).append((cname, box))
    return ann


def match_detections_to_gt(gt, dets, iou_thr=0.5):
    tp = fp = fn = 0

    used_gt = [False] * len(gt)

    for det in dets:
        _, cname, conf, x1, y1, x2, y2 = det
        dbox = (x1, y1, x2, y2)

        best_iou = 0
        best_id = -1

        for i, (_, gt_box) in enumerate(gt):
            if used_gt[i]:
                continue

            cur = iou(dbox, gt_box)
            if cur > best_iou:
                best_iou = cur
                best_id = i

        if best_iou >= iou_thr:
            tp += 1
            used_gt[best_id] = True
        else:
            fp += 1

    for used in used_gt:
        if not used:
            fn += 1

    return tp, fp, fn


def compute_tpr_fdr(tp: int, fp: int, fn: int):
    tpr = tp / (tp + fn) if tp + fn > 0 else 0
    fdr = fp / (tp + fp) if tp + fp > 0 else 0
    return tpr, fdr
