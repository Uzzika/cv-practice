import numpy as np
from typing import List, Tuple
from base_struct import BaseDetector, Detection


class FasterRCNNDetector(BaseDetector):
    def postprocess(self, outputs, image_shape):
        detections = []
        img_height, img_width = image_shape

        for detection in outputs[0, 0]:
            confidence = float(detection[2])
            class_id = int(detection[1])
        
            if class_id == 2 and confidence > self.config.confidence_threshold:
                x1 = int(detection[3] * img_width)
                y1 = int(detection[4] * img_height)
                x2 = int(detection[5] * img_width)
                y2 = int(detection[6] * img_height)

                if x2 > x1 and y2 > y1:
                    detections.append(Detection(
                        class_id=class_id,
                        class_name="car",
                        confidence=confidence,
                        bbox=(x1, y1, x2, y2)
                    ))

        return detections
