from typing import List, Tuple
from base_struct import Detection


class DetectionEvaluator:
    def __init__(self, target_class = "car"):
        self.target_class = target_class
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0
        
    def calculate_iou(self, box1, box2):
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        intersection_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0
    
    def evaluate_frame(self, detections, ground_truth, iou_threshold = 0.5):
        frame_tp = 0
        frame_fp = 0
        frame_fn = 0
        
        gt_used = [False] * len(ground_truth)
        
        detections.sort(key=lambda x: x.confidence, reverse=True)
        
        for detection in detections:
            if detection.class_name.lower() != self.target_class.lower():
                continue
                
            best_iou = 0
            best_gt_idx = -1
            
            for i, gt_box in enumerate(ground_truth):
                if not gt_used[i]:
                    iou = self.calculate_iou(detection.bbox, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = i
            
            if best_iou >= iou_threshold and best_gt_idx != -1:
                frame_tp += 1
                self.true_positives += 1
                gt_used[best_gt_idx] = True
            else:
                frame_fp += 1
                self.false_positives += 1
        
        unused_gt = sum(1 for used in gt_used if not used)
        frame_fn += unused_gt
        self.false_negatives += unused_gt
        
        frame_tpr = frame_tp / (frame_tp + frame_fn) if (frame_tp + frame_fn) > 0 else 0
        frame_fdr = frame_fp / (frame_tp + frame_fp) if (frame_tp + frame_fp) > 0 else 0
        
        return frame_tpr, frame_fdr
    
    def get_metrics(self):
        tpr = self.true_positives / (self.true_positives + self.false_negatives) if (self.true_positives + self.false_negatives) > 0 else 0
        fdr = self.false_positives / (self.true_positives + self.false_positives) if (self.true_positives + self.false_positives) > 0 else 0
        return tpr, fdr
    
    def reset(self):
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0
