from detectors import BaseDetector
import cv2
import numpy as np
class NanoDetDetector(BaseDetector):
    def __init__(self, model, conf_th, nms_th):
        super().__init__(model, (416,416))
        self.width = 416
        self.height = 416
        self.net = cv2.dnn.readNet(model)
        self.strides = [8,16,32,64]
        self.num_classes = 80
        self.grid = np.arange(8)
        self.conf_treshold = conf_th
        self.nms_treshold = nms_th
    def decode_boxes(self, boxes_raw, grid_x, grid_y, pitch):
        boxes = []
        for i in range(4):  # x1, y1, x2, y2
            exp = np.exp(boxes_raw[:, 8*i:8*(i+1)])
            wing = np.sum(exp * self.grid, axis=1) / np.sum(exp, axis=1) * pitch
            if i == 0:
                coord = np.maximum(grid_x - wing, 0)  # x1
            elif i == 1:
                coord = np.maximum(grid_y - wing, 0)  # y1
            elif i == 2:
                coord = np.minimum(grid_x + wing, 1)  # x2
            else:
                coord = np.minimum(grid_y + wing, 1)  # y2
            boxes.append(coord.reshape(-1, 1))
        return np.concatenate(boxes, axis=1)
    def preprocess(self, img):
        blob = cv2.dnn.blobFromImage(
            img, 1/57.375, self.input_size,
            (103.53,116.28,123.675), swapRB=False
        )
        return blob

    def postprocess(self, outputs, img):
        outputs = outputs[0]
        H, W = img.shape[:2]
        start = 0
        boxes_list, scores_list = [], []
        for stride in self.strides:
            fh = self.height // stride
            fw = self.width // stride
            num_preds = fh * fw

            preds_level = outputs[start:start+num_preds]
            scores_level = preds_level[:, :self.num_classes]
            boxes_level = preds_level[:, self.num_classes:]

            x = np.arange(fw) + 0.5
            y = np.arange(fh) + 0.5
            gx, gy = np.meshgrid(x, y)
            gx = gx.flatten() / fw
            gy = gy.flatten() / fh
            pitch = stride / self.width
            boxes_decoded = self.decode_boxes(boxes_level, gx, gy, pitch)

            boxes_list.append(boxes_decoded)
            scores_list.append(scores_level)

            start += num_preds
        boxes = np.concatenate(boxes_list, axis=0)
        scores = np.concatenate(scores_list, axis=0)
        class_ids = np.argmax(scores, axis=1)
        confs = scores[np.arange(len(scores)), class_ids]
        mask = confs >= self.conf_treshold
        boxes = boxes[mask]
        confs = confs[mask]
        class_ids = class_ids[mask]

        bboxes_xyxy = []
        for x1, y1, x2, y2 in boxes:
            x1_px = int(x1 * W)
            y1_px = int(y1 * H)
            x2_px = int(x2 * W)
            y2_px = int(y2 * H)
            bboxes_xyxy.append([x1_px, y1_px, x2_px, y2_px])

        nms_indices = cv2.dnn.NMSBoxes(
            bboxes=[[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in bboxes_xyxy],
            scores=confs.tolist(),
            score_threshold=self.conf_treshold,   
            nms_threshold=self.nms_treshold
        )
        results = []
        if len(nms_indices) > 0:
            nms_indices = nms_indices.flatten()
            for i in nms_indices:
                x1, y1, x2, y2 = boxes[i]
                x1_px = int(x1 * W)
                y1_px = int(y1 * H)
                x2_px = int(x2 * W)
                y2_px = int(y2 * H)
                results.append((x1_px,y1_px,x2_px-x1_px,y2_px-y1_px, confs[i], class_ids[i]))
        return results