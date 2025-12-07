from typing import Dict
from base_struct import ModelConfig
from yolo_detector import YOLODetector
from mobilenet_detector import SSDMobileNetDetector
from rcnn_detector import FasterRCNNDetector

class VehicleDetectorFactory:   
    @staticmethod
    def get_available_models():
        return {
            "yolo": {
                "name": "YOLOv4",
                "description": "YOLOv4 trained on COCO dataset",
                "vehicle_classes": ["car", "truck", "bus", "motorcycle"]
            },
            "mobilenet": {
                "name": "SSD MobileNet V3",
                "description": "SSD with MobileNet V3 backbone",
                "vehicle_classes": ["car", "truck", "bus"]
            },
            "rcnn": {
                "name": "Faster R-CNN",
                "description": "Faster R-CNN with Inception V2",
                "vehicle_classes": ["car", "truck", "bus"]
            }
        }
    
    @staticmethod
    def create_detector(model_type, confidence_threshold = 0.5):
        models_configs = {
            "yolo": ModelConfig(
                name="YOLOv4",
                model_path="models/yolov4.weights",
                config_path="configs/yolov4.cfg",
                classes=["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"],
                input_size=(416, 416),
                scale_factor=1/255.0,
                swap_rb=True,
                confidence_threshold=confidence_threshold,
                nms_threshold=0.4
            ),
            "mobilenet": ModelConfig(
                name="SSD MobileNet V3",
                model_path="models/mobilenet_frozen_inference_graph.pb",
                config_path="configs/ssd_mobilenet_v2_coco_2018_03_29.pbtxt",
                classes=["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"],
                input_size=(600, 600),
                scale_factor=1.0,
                swap_rb=True,
                confidence_threshold=confidence_threshold,
                nms_threshold=0.4
            ),
            "rcnn": ModelConfig(
                name="Faster R-CNN",
                model_path="models/rcnn_frozen_inference_graph.pb",
                config_path="configs/faster_rcnn_inception_v2_coco_2018_01_28.pbtxt",
                classes=["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"],
                input_size=(300, 300),
                scale_factor=1.0,
                swap_rb=True,
                confidence_threshold=confidence_threshold,
                nms_threshold=0.4
            )
        }
        
        if model_type not in models_configs:
            raise ValueError(f"Unknown model type: {model_type}")
        
        config = models_configs[model_type]
        
        if model_type == "yolo":
            return YOLODetector(config)
        elif model_type == "mobilenet":
            return SSDMobileNetDetector(config)
        elif model_type == "rcnn":
            return FasterRCNNDetector(config)
