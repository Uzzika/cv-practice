import os
import argparse
import cv2
import numpy as np

from detectors import create_detector
from metrics import load_annotations_txt, match_detections_to_gt, compute_tpr_fdr


def list_image_files(frames_dir):
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    files = [os.path.join(frames_dir, f)
             for f in os.listdir(frames_dir)
             if f.lower().endswith(exts)]
    files.sort()
    return files


def get_color(name):
    colors = {
        "car": (0, 255, 0),
        "truck": (0, 0, 255),
        "bus": (255, 0, 0),
        "motorbike": (0, 255, 255),
        "bicycle": (255, 0, 255),
        "train": (255, 255, 0),
    }
    return colors.get(name, (200, 200, 200))


def draw(image, detections, gt_boxes, matches):
    match_dict = {d: g for d, g in matches}

    for det_idx, det in enumerate(detections):
        class_id, cname, conf, x1, y1, x2, y2 = det
        color = get_color(cname)

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # предсказанный класс + confidence (3 знака)
        cv2.putText(image, f"{cname} {conf:.3f}",
                    (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # наблюдаемый (GT) класс — над прямоугольником
        if det_idx in match_dict:
            gt_class = gt_boxes[match_dict[det_idx]][0]
            gt_text = f"GT: {gt_class}"
        else:
            gt_text = "GT: none"

        cv2.putText(image, gt_text,
                    (x1, max(0, y1 - 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return image


def main():
    parser = argparse.ArgumentParser(
        description="Практическая работа №2. Детектирование объектов с OpenCV DNN."
    )
    parser.add_argument("--frames_dir", default="./data/imgs_MOV03478",
                        help="Путь к кадрам видео")
    parser.add_argument("--annotations", default="./data/mov03478.txt",
                        help="Путь к разметке")
    parser.add_argument("--model",
                        default="yolov3",
                        choices=["yolov3", "yolov4-tiny", "ssd"],
                        help="Модель детектора")
    parser.add_argument("--show", action="store_true",
                        help="Показывать окна с детекцией")
    parser.add_argument("--max_frames", type=int, default=-1,
                        help="Максимум кадров (-1 = все)")
    args = parser.parse_args()

    # Пути к моделям
    if args.model == "yolov3":
        model_paths = {
            "cfg": "./models/yolov3.cfg",
            "weights": "./models/yolov3.weights",
        }
    elif args.model == "yolov4-tiny":
        model_paths = {
            "cfg": "./models/yolov4-tiny.cfg",
            "weights": "./models/yolov4-tiny.weights",
        }
    elif args.model == "ssd":
        model_paths = {
            "proto": "./models/deploy.prototxt",
            "model": "./models/mobilenet_iter_73000.caffemodel",
        }
    else:
        raise ValueError("Unknown model")

    print(f"[INFO] Using model: {args.model}")
    detector = create_detector(args.model, model_paths)

    annotations = load_annotations_txt(args.annotations)
    frames = list_image_files(args.frames_dir)

    total_tp = total_fp = total_fn = 0

    for idx, fpath in enumerate(frames):
        if 0 <= args.max_frames <= idx:
            break

        img = cv2.imread(fpath)
        if img is None:
            print(f"[WARN] Cannot read {fpath}")
            continue

        detections = detector.detect(img)
        frame_name = os.path.splitext(os.path.basename(fpath))[0]
        try:
            frame_id = int(frame_name)
        except:
            frame_id = idx  # fallback

        gt_boxes = annotations.get(frame_id, [])

        tp, fp, fn, matches = match_detections_to_gt(detections, gt_boxes)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        print(f"[{idx}] det={len(detections)}, gt={len(gt_boxes)}, TP={tp}, FP={fp}, FN={fn}")

        if args.show:
            vis = img.copy()
            vis = draw(vis, detections, gt_boxes, matches)
            cv2.imshow(f"Detections ({args.model})", vis)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cv2.destroyAllWindows()

    tpr, fdr = compute_tpr_fdr(total_tp, total_fp, total_fn)
    print("\n========== SUMMARY ==========")
    print(f"Model: {args.model}")
    print(f"TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")
    print(f"TPR = {tpr:.4f}")
    print(f"FDR = {fdr:.4f}")
    print("================================")


if __name__ == "__main__":
    main()
