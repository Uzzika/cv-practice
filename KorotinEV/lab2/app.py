import cv2
import numpy as np
import os
import glob
import time
from typing import List, Tuple
from base_struct import Detection
from detector_factory import VehicleDetectorFactory
from evaluator import DetectionEvaluator
from annotation_loader import AnnotationLoader


class VehicleDetectionApp:
    def __init__(self):
        self.detection_colors = (0, 255, 0)     
        self.ground_truth_color = (255, 0, 255)
    
    def draw_detections(self, image, detections):
        result = image.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            class_name = detection.class_name.lower()
            color = self.detection_colors
            confidence = detection.confidence
            
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
            
            label = f"{class_name}: {confidence:.3f}"
            
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            cv2.rectangle(
                result, 
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                color,
                -1
            )
            
            cv2.putText(
                result,
                label,
                (x1, y1 - baseline - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
        
        return result
    
    def draw_ground_truth(self, image, ground_truth):
        result = image.copy()
        
        for i, bbox in enumerate(ground_truth):
            x1, y1, x2, y2 = bbox
            
            cv2.rectangle(result, (x1, y1), (x2, y2), self.ground_truth_color, 2)
            
            label = "GT"
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            text_x = x2 - text_width
            text_y = y1 + text_height + baseline + 5
            
            cv2.rectangle(
                result,
                (text_x, text_y - text_height - baseline - 5),
                (text_x + text_width, text_y),
                self.ground_truth_color,
                -1
            )
            
            cv2.putText(
                result,
                label,
                (text_x, text_y - baseline - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
        
        return result
    
    def display_metrics(self, image, tpr, fdr, frame_tpr = None, frame_fdr = None):
        result = image.copy()
        
        metrics_text = [
            f"TPR: {tpr:.3f}",
            f"FDR: {fdr:.3f}"
        ]
        
        if frame_tpr is not None and frame_fdr is not None:
            metrics_text.extend([
                f"Frame TPR: {frame_tpr:.3f}",
                f"Frame FDR: {frame_fdr:.3f}"
            ])
        
        for i, text in enumerate(metrics_text):
            cv2.putText(
                result,
                text,
                (10, 30 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
        
        return result
    
    def run(self, images_path: str, annotation_path: str, model_type: str, 
            confidence_threshold: float = 0.5, display: bool = True, 
            show_ground_truth: bool = False):
        
        available_models = VehicleDetectorFactory.get_available_models()
        if model_type not in available_models:
            print(f"Доступные модели: {list(available_models.keys())}")
            return
        
        print(f"Выбрана модель: {available_models[model_type]['name']}")
        print(f"Классы транспортных средств: {available_models[model_type]['vehicle_classes']}")
        print(f"Отображение Ground Truth: {'ВКЛ' if show_ground_truth else 'ВЫКЛ'}")
        
        try:
            detector = VehicleDetectorFactory.create_detector(model_type, confidence_threshold)
        except Exception as e:
            print(f"Ошибка при создании детектора: {e}")
            print("Убедитесь, что модели загружены в папку 'models/'")
            return

        annotation_loader = AnnotationLoader(annotation_path)
        evaluator = DetectionEvaluator(target_class="car")
        
        image_files = sorted(glob.glob(os.path.join(images_path, "*.jpg")))
        print(f"Найдено изображений: {len(image_files)}")
        
        total_time = 0
        processed_frames = 0
        
        for image_file in image_files:
            image_id = os.path.splitext(os.path.basename(image_file))[0]
            
            image = cv2.imread(image_file)
            if image is None:
                print(f"Не удалось загрузить изображение: {image_file}")
                continue
            
            ground_truth = annotation_loader.get_ground_truth(image_id)
            
            start_time = time.time()
            detections = detector.detect(image)
            end_time = time.time()
            
            processing_time = end_time - start_time
            total_time += processing_time
            processed_frames += 1
            
            frame_tpr, frame_fdr = evaluator.evaluate_frame(detections, ground_truth)
            
            overall_tpr, overall_fdr = evaluator.get_metrics()
            
            if display:
                result_image = image.copy()
                
                if show_ground_truth:
                    result_image = self.draw_ground_truth(result_image, ground_truth)
                
                result_image = self.draw_detections(result_image, detections)
                
                result_image = self.display_metrics(result_image, overall_tpr, overall_fdr, frame_tpr, frame_fdr)
                
                cv2.imshow("Vehicle Detection", result_image)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
            
            print(f"Обработано: {image_id}.jpg | Время: {processing_time:.3f}s | "
                  f"TPR: {overall_tpr:.3f} | FDR: {overall_fdr:.3f} | "
                  f"Frame TPR: {frame_tpr:.3f} | Frame FDR: {frame_fdr:.3f}")
        
        final_tpr, final_fdr = evaluator.get_metrics()
        avg_time = total_time / processed_frames if processed_frames > 0 else 0
        
        print("\n" + "="*50)
        print("ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ:")
        print(f"Модель: {available_models[model_type]['name']}")
        print(f"Обработано кадров: {processed_frames}")
        print(f"Среднее время обработки: {avg_time:.3f}s")
        print(f"TPR (True Positive Rate): {final_tpr:.3f}")
        print(f"FDR (False Discovery Rate): {final_fdr:.3f}")
        print(f"True Positives: {evaluator.true_positives}")
        print(f"False Positives: {evaluator.false_positives}")
        print(f"False Negatives: {evaluator.false_negatives}")
        
        if display:
            cv2.destroyAllWindows()
