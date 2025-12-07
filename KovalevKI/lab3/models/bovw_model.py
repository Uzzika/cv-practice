import cv2
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
from typing import List, Tuple


class BoVWClassifier:
    def __init__(self, n_clusters: int = 100, detector_name: str = 'SIFT'):
        self.n_clusters = n_clusters
        self.detector_name = detector_name
        self.detector, self.descriptor = self._init_feature_extractor()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.classifier = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel='rbf', C=10, gamma='scale', random_state=42))
        ])
        self.is_fitted = False

    def _init_feature_extractor(self):
        if self.detector_name == 'SIFT':
            return cv2.SIFT_create(), cv2.SIFT_create()
        elif self.detector_name == 'ORB':
            orb = cv2.ORB_create(nfeatures=500)
            return orb, orb
        elif self.detector_name == 'AKAZE':
            return cv2.AKAZE_create(), cv2.AKAZE_create()

    def _extract_features(self, image_paths: List[str]) -> Tuple[np.ndarray, List[np.ndarray]]:
        all_descriptors = []
        per_img_desc = []

        for path in image_paths:
            try:
                img = cv2.imread(path)
                if img is None:
                    desc_dim = 128 if self.detector_name == 'SIFT' else 32
                    per_img_desc.append(np.zeros((1, desc_dim), dtype=np.float32))
                    continue

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                kp, des = self.descriptor.detectAndCompute(gray, None)

                if des is None or len(des) == 0:
                    desc_dim = self.descriptor.descriptorSize()
                    des = np.zeros((1, desc_dim), dtype=np.float32)

                des = des.astype(np.float32)
                all_descriptors.append(des)
                per_img_desc.append(des)

            except Exception as e:
                print(f"Ошибка при обработке {path}: {e}")
                desc_dim = self.descriptor.descriptorSize()
                per_img_desc.append(np.zeros((1, desc_dim), dtype=np.float32))

        if all_descriptors:
            all_descriptors = np.vstack(all_descriptors)
        else:
            desc_dim = self.descriptor.descriptorSize()
            all_descriptors = np.empty((0, desc_dim))

        return all_descriptors, per_img_desc
        
    def visualize_keypoints(self, image_path: str, save_to: str = "keypoints.jpg", max_points: int = 1000000):
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Не удалось загрузить изображение: {image_path}")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, _ = self.detector.detectAndCompute(gray, None)

        kp = kp[:max_points]

        img_kp = cv2.drawKeypoints(
            img, kp,
            outImage=None,
            color=(0, 255, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_kp, f"{self.detector_name} keypoints: {len(kp)}", 
                    (10, 30), font, 0.8, (0, 0, 255), 2)

        cv2.imwrite(save_to, img_kp)
        print(f"Ключевые точки сохранены: {os.path.abspath(save_to)}")
            
    
    def fit(self, image_paths: List[str], labels: List[int]):
        print(f"BoVW: извлечение дескрипторов ({len(image_paths)} изображений)...")
        all_desc, _ = self._extract_features(image_paths)
        print(f"Всего дескрипторов: {len(all_desc)}")

        print(f"BoVW: обучение KMeans (n_clusters={self.n_clusters})...")
        self.kmeans.fit(all_desc)

        print("BoVW: построение гистограмм...")
        X_hist = self._images_to_histograms(image_paths)
        print(f"BoVW: обучение SVM на {X_hist.shape} признаках...")
        self.classifier.fit(X_hist, labels)
        self.is_fitted = True
        print("BoVW обучен.")

    def _images_to_histograms(self, image_paths: List[str]) -> np.ndarray:
        _, per_img_desc = self._extract_features(image_paths)
        histograms = []
        for des in per_img_desc:
            if des.shape[0] == 0:
                hist = np.zeros(self.n_clusters)
            else:
                words = self.kmeans.predict(des)
                hist, _ = np.histogram(words, bins=np.arange(self.n_clusters + 1), density=True)
            histograms.append(hist)
        return np.array(histograms, dtype=np.float32)

    def predict(self, image_paths: List[str]) -> np.ndarray:
        X_hist = self._images_to_histograms(image_paths)
        return self.classifier.predict(X_hist)

    def score(self, image_paths: List[str], labels: List[int]) -> float:
        preds = self.predict(image_paths)
        return float(np.mean(preds == labels))

    def save(self, path: str):
        data = {
            'detector_name': self.detector_name,
            'n_clusters': self.n_clusters,
            'kmeans': self.kmeans,
            'classifier': self.classifier,
            'is_fitted': self.is_fitted
        }
        joblib.dump(data, path)
        print(f"BoVW сохранён в: {path}")

    @classmethod
    def load(cls, path: str):
        data = joblib.load(path)
        obj = cls(n_clusters=data['n_clusters'], detector_name=data['detector_name'])
        obj.kmeans = data['kmeans']
        obj.classifier = data['classifier']
        obj.is_fitted = data['is_fitted']
        return obj