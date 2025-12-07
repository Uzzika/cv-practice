from detectors import BaseDetector
import cv2
import numpy as np
class YOLOv8Detector(BaseDetector):
    def __init__(self, model_path, conf_th, nms_th):
        super().__init__(model_path, (640, 640))
        self.net = cv2.dnn.readNet(model_path)
        self.conf_th = conf_th
        self.nms_th = nms_th

    def preprocess(self, img):
        blob = cv2.dnn.blobFromImage(
            img, 1/255.0, size = self.input_size, swapRB=True, crop=False
        )
        return blob

    def postprocess(self, outputs, img):
        outputs = outputs[0].T
        results = []
        
        boxes = []
        confs = []
        class_ids = []

        H, W = img.shape[:2]
        for i in range(outputs.shape[0]):
            bbox_data = outputs[i, 0:4]
            class_scores = outputs[i, 4:]
            class_id = np.argmax(class_scores)
            score = class_scores[class_id]

            if score > self.conf_th:
                x_center, y_center, w, h = bbox_data
                
                x = int((x_center - w / 2))
                y = int((y_center - h / 2))
                w = int(w)
                h = int(h)
                
                boxes.append([x, y, w, h])
                confs.append(float(score))
                class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confs, self.conf_th, self.nms_th)

        if len(indices) > 0:
            indices = indices.flatten()

            for i in indices:
                box = boxes[i]
                x, y, w, h = box

                x_scaled = int(x * W / 640)
                y_scaled = int(y * H / 640)
                w_scaled = int(w * W / 640)
                h_scaled = int(h * H / 640)
                results.append((x_scaled,y_scaled,w_scaled,h_scaled, confs[i], class_ids[i]))

        return results
    
    
