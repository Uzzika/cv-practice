from detectors import BaseDetector
import cv2
import numpy as np
class MobileNetDetector(BaseDetector):
    def __init__(self, pb, pbtxt, conf_th, nms_th):
        super().__init__(pb, (600, 600))
        self.net = cv2.dnn_DetectionModel(pb, pbtxt)
        self.net.setInputSwapRB(True)
        self.net.setInputSize(600, 600)
        self.conf_th = conf_th
        self.nms_th = nms_th
    def preprocess(self, img):
        return img
    def postprocess(self, outputs):
        return outputs
    def detect(self, img):
        classIds, confs, boxes = self.net.detect(
            img,
            confThreshold=self.conf_th,
            nmsThreshold=self.nms_th
        )

        if len(classIds)==0 or len(confs)==0 or len(boxes)==0:
            return []

        classIds = classIds.flatten()
        confs = confs.flatten()

        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            confs.tolist(),
            self.conf_th,
            self.nms_th
        )

        results = []

        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                cls = int(classIds[i]-1)
                conf = float(confs[i])
                results.append((x, y, w, h, conf, cls))

        return results

