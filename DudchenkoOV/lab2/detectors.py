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


# ====================== YOLOv3 Detector ======================

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
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        input_size: Tuple[int, int] = (416, 416),
    ):
        super().__init__(class_names, vehicle_classes)
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size

        # Load YOLOv3 network
        self.net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

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
            scalefactor=1/255.0,
            size=self.input_size,
            swapRB=True,
            crop=False
        )
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)

        (H, W) = image.shape[:2]
        boxes = []
        confidences = []
        class_ids = []

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


# ============ Detector Factory ============

def create_detector(model_name: str, model_paths):
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

    vehicle_classes = ['car', 'bus', 'truck', 'motorbike', 'bicycle', 'train']

    if model_name == "yolov3":
        return YOLOv3Detector(
            cfg_path=model_paths["cfg"],
            weights_path=model_paths["weights"],
            class_names=coco_classes,
            vehicle_classes=vehicle_classes,
        )

    raise ValueError("Unknown model name")
