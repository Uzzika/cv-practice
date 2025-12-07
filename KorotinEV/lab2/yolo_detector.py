import numpy as np
from typing import List, Tuple
from base_struct import BaseDetector, Detection

class YOLODetector(BaseDetector):
    def postprocess(self, outputs, image_shape):
        detections = []
        img_height, img_width = image_shape
        
        for detection in outputs:
            try:
                object_confidence = detection[4]
                
                if object_confidence < 0.1:
                    continue
                
                class_probs = detection[5:85]
                class_id = np.argmax(class_probs)
                class_confidence = class_probs[class_id]
                
                total_confidence = object_confidence * class_confidence
                
                if class_id == 2 and total_confidence > self.config.confidence_threshold:
                    center_x = detection[0] * img_width
                    center_y = detection[1] * img_height
                    width = detection[2] * img_width
                    height = detection[3] * img_height
                    
                    x1 = int(center_x - width / 2)
                    y1 = int(center_y - height / 2)
                    x2 = int(x1 + width)
                    y2 = int(y1 + height)
                    
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(img_width - 1, x2)
                    y2 = min(img_height - 1, y2)
                    
                    if x2 > x1 and y2 > y1:
                            detections.append(Detection(
                            class_id=class_id,
                            class_name="car",
                            confidence=float(total_confidence),
                            bbox=(x1, y1, x2, y2)
                        ))
                        
            except Exception as e:
                print(f"ERROR processing detection: {e}")
                continue
        
        return detections
