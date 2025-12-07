import numpy as np
import cv2
import os
def load_images_from_split(file, mode = None):
    images, labels= [], []
    class_dirs = {} 

    with open(file, 'r') as f:
        for line in f:
            path = line.strip()
            if path:
                class_dir = os.path.basename(os.path.dirname(path))
                if class_dir not in class_dirs:
                    class_dirs[class_dir] = len(class_dirs)

    with open(file, 'r') as f:
        for line in f:
            path = line.strip()
            path = "data\\" + path
            if path:
                img = cv2.imread(path)
                if img is not None:
                    if mode == "NN": img = cv2.resize(img,(224,224))
                    class_dir = os.path.basename(os.path.dirname(path))
                    images.append(img)
                    labels.append(class_dirs[class_dir])

    return images, np.array(labels)