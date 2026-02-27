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


def match_detections_to_gt(detections, gt_boxes, iou_thr=0.3):
    """
    detections: [(class_id, class_name, conf, x1, y1, x2, y2)]
    gt_boxes:   [(class_name, (x1, y1, x2, y2))]

    return: tp, fp, fn, matches[(det_index, gt_index)]
    """
    matched_gt = set()
    matches = []
    tp = 0
    fp = 0

    for det_idx, det in enumerate(detections):
        det_class = det[1].strip().lower()
        det_box = (det[3], det[4], det[5], det[6])

        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx, (gt_class, gt_box) in enumerate(gt_boxes):
            gt_class = gt_class.strip().lower()
            if gt_idx in matched_gt:
                continue
            # нормализация
            det_class = det_class.strip().lower()
            gt_class = gt_class.strip().lower()

            # допускаем частичное совпадение
            if det_class not in gt_class and gt_class not in det_class:
                continue

            iou_val = iou(det_box, gt_box)  # <-- правильная функция
            if iou_val > best_iou:
                best_iou = iou_val
                best_gt_idx = gt_idx

        if best_iou >= iou_thr:
            tp += 1
            matched_gt.add(best_gt_idx)
            matches.append((det_idx, best_gt_idx))
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn, matches


def compute_tpr_fdr(tp: int, fp: int, fn: int):
    tpr = tp / (tp + fn) if tp + fn > 0 else 0
    fdr = fp / (tp + fp) if tp + fp > 0 else 0
    return tpr, fdr
