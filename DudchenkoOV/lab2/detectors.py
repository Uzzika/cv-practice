import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple

# Тип выхода детектора: (class_id, class_name, confidence, x1, y1, x2, y2)
Detection = Tuple[int, str, float, int, int, int, int]


class BaseDetector(ABC):
    def __init__(self, class_names: List[str], vehicle_classes: List[str]):
        self.class_names = class_names
        self.vehicle_classes = set(vehicle_classes)

    @abstractmethod
    def detect(self, image: np.ndarray) -> List[Detection]:
        pass


class YOLOv3Detector(BaseDetector):
    """
    YOLOv3 COCO detector using OpenCV DNN.
    """

    def __init__(
        self,
        cfg_path: str,
        weights_path: str,
        class_names: List[str],
        vehicle_classes: List[str],
        conf_threshold: float = 0.3,
        nms_threshold: float = 0.4,
        input_size: Tuple[int, int] = (416, 416),
    ):
        super().__init__(class_names, vehicle_classes)
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size

        # Load YOLO network
        self.net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)

        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        # отключаем Winograd (фикс OpenCV 4.7)
        self.net.enableWinograd(False)

        # Output layers – support for new/old OpenCV
        try:
            self.output_layers = self.net.getUnconnectedOutLayersNames()
        except AttributeError:
            layer_names = self.net.getLayerNames()
            unconnected = self.net.getUnconnectedOutLayers()
            indices = []
            for i in unconnected:
                if isinstance(i, (list, tuple, np.ndarray)):
                    indices.append(int(i[0]) - 1)
                else:
                    indices.append(int(i) - 1)
            self.output_layers = [layer_names[idx] for idx in indices]

    def detect(self, image: np.ndarray) -> List[Detection]:
        blob = cv2.dnn.blobFromImage(
            image,
            scalefactor=1 / 255.0,
            size=self.input_size,
            swapRB=True,
            crop=False,
        )
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)

        (H, W) = image.shape[:2]
        boxes: List[List[int]] = []
        confidences: List[float] = []
        class_ids: List[int] = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                confidence = float(scores[class_id])

                if confidence < self.conf_threshold:
                    continue

                class_name = self.class_names[class_id]
                if class_name not in self.vehicle_classes:
                    continue

                center_x = int(detection[0] * W)
                center_y = int(detection[1] * H)
                w = int(detection[2] * W)
                h = int(detection[3] * H)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                x = max(0, x)
                y = max(0, y)
                w = max(0, w)
                h = max(0, h)

                boxes.append([x, y, w, h])
                confidences.append(confidence)
                class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(
            boxes, confidences,
            self.conf_threshold, self.nms_threshold
        )

        detections: List[Detection] = []

        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                detections.append((
                    class_ids[i],
                    self.class_names[class_ids[i]],
                    confidences[i],
                    x, y, x + w, y + h
                ))

        return detections


class YOLOv4TinyDetector(YOLOv3Detector):
    """
    YOLOv4-tiny в OpenCV настраивается так же, как YOLOv3:
    тот же формат cfg/weights и те же выходы.
    Просто используем другой cfg/weights.
    """
    pass


def load_class_names_from_file(path: str) -> List[str]:
    """
    Читает список классов из текстового файла (по одному имени в строке).
    """
    names: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if name:
                names.append(name)
    return names


class SSDCaffeDetector(BaseDetector):
    """
    SSD MobileNet (Caffe):
      - deploy.prototxt
      - mobilenet_iter_73000.caffemodel
      - ssd_classes.txt
    Используется стандартный интерфейс cv2.dnn.readNetFromCaffe.
    """

    def __init__(
        self,
        proto_path: str,
        model_path: str,
        class_names: List[str],
        vehicle_classes: List[str],
        conf_threshold: float = 0.5,
        input_size: Tuple[int, int] = (300, 300),
    ):
        super().__init__(class_names, vehicle_classes)
        self.conf_threshold = conf_threshold
        self.input_size = input_size

        self.net = cv2.dnn.readNetFromCaffe(proto_path, model_path)

    def detect(self, image: np.ndarray) -> List[Detection]:
        (h, w) = image.shape[:2]

        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, self.input_size),
            scalefactor=0.007843,  # 1/127.5
            size=self.input_size,
            mean=127.5
        )

        self.net.setInput(blob)
        detections_raw = self.net.forward()  # [1, 1, N, 7]

        detections: List[Detection] = []

        for i in range(detections_raw.shape[2]):
            conf = float(detections_raw[0, 0, i, 2])
            if conf < self.conf_threshold:
                continue

            class_id = int(detections_raw[0, 0, i, 1])
            if class_id < 0 or class_id >= len(self.class_names):
                continue

            class_name = self.class_names[class_id]

            # оставляем только транспорт
            if class_name not in self.vehicle_classes:
                continue

            box = detections_raw[0, 0, i, 3:7]
            x1 = int(box[0] * w)
            y1 = int(box[1] * h)
            x2 = int(box[2] * w)
            y2 = int(box[3] * h)

            detections.append((class_id, class_name, conf, x1, y1, x2, y2))

        return detections


def create_detector(model_name: str, model_paths):
    # COCO классы
    coco_classes = [
        'person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
        'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
        'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
        'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
        'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa',
        'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
        'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    # классы транспорта
    vehicle_classes = ['car', 'bus', 'truck', 'motorbike', 'bicycle', 'train']

    if model_name == "yolov3":
        return YOLOv3Detector(
            cfg_path=model_paths["cfg"],
            weights_path=model_paths["weights"],
            class_names=coco_classes,
            vehicle_classes=vehicle_classes,
        )
    elif model_name == "yolov4-tiny":
        return YOLOv4TinyDetector(
            cfg_path=model_paths["cfg"],
            weights_path=model_paths["weights"],
            class_names=coco_classes,
            vehicle_classes=vehicle_classes,
        )
    elif model_name == "ssd":
        ssd_classes = load_class_names_from_file("./models/ssd_classes.txt")
        return SSDCaffeDetector(
            proto_path=model_paths["proto"],
            model_path=model_paths["model"],
            class_names=ssd_classes,
            vehicle_classes=vehicle_classes,
            conf_threshold=0.5,
        )

    raise ValueError("Unknown model name")
