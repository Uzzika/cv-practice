import os
import cv2
import numpy as np
import joblib
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from base_classifier import BaseClassifier

class BOWClassifier(BaseClassifier):
    def __init__(self, n_clusters=100, image_size=(224, 224), class_names=None):
        super().__init__('bow', image_size, class_names)
        self.n_clusters = n_clusters
        self.kmeans = None
        self.svm = None
        self.scaler = StandardScaler()
        self.sift = cv2.SIFT_create()
        
    def extract_features(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Ошибка загрузки изображения: {image_path}")
            return None
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        return descriptors
    
    def build_vocabulary(self, all_descriptors):
        all_descriptors = np.vstack(all_descriptors)
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.kmeans.fit(all_descriptors)
        return self.kmeans
    
    def descr_to_histogram(self, descriptors):
        labels = self.kmeans.predict(descriptors)
        histogram, _ = np.histogram(labels, bins=self.n_clusters, range=(0, self.n_clusters))
        histogram = histogram.astype(float)
        if np.sum(histogram) > 0:
            histogram /= np.sum(histogram)
        return histogram

    def train(self, train_file, images_dir="."):
        from sklearn.metrics import accuracy_score
        
        train_paths, train_labels, label_ids = self.load_data(train_file, images_dir)
        
        if len(train_paths) == 0:
            print("Ошибка: нет данных для обучения!")
            return 0
        
        all_descriptors = []
        
        for image_path in train_paths:
            descriptors = self.extract_features(image_path)
            if descriptors is not None:
                all_descriptors.append(descriptors)
        
        self.build_vocabulary(all_descriptors)
 
        train_features = []
        for descriptors in all_descriptors:
            histogram = self.descr_to_histogram(descriptors)
            train_features.append(histogram)
        
        train_features = np.array(train_features)     
        train_features_scaled = self.scaler.fit_transform(train_features)

        self.svm = SVC(kernel='linear', probability=True, random_state=42)
        self.svm.fit(train_features_scaled, label_ids)

        train_predictions = self.svm.predict(train_features_scaled)
        train_accuracy = accuracy_score(label_ids, train_predictions)
        
        print(f"Точность на обучающей выборке: {train_accuracy:.4f}")
        return train_accuracy

    def test(self, test_file, images_dir="."):
        from sklearn.metrics import accuracy_score, classification_report
        
        test_paths, test_labels, label_ids = self.load_data(test_file, images_dir)
        
        if len(test_paths) == 0:
            print("Ошибка: нет данных для тестирования!")
            return 0
        
        test_features = []
        
        for path in test_paths:
            descriptors = self.extract_features(path)
            if descriptors is not None:
                histogram = self.descr_to_histogram(descriptors)
                test_features.append(histogram)
        
        test_features = np.array(test_features)
        test_features_scaled = self.scaler.transform(test_features)
        
        test_predictions = self.svm.predict(test_features_scaled)
        test_accuracy = accuracy_score(label_ids, test_predictions)
        
        test_predictions_labels = [self.class_names[pred] for pred in test_predictions]
        
        print(f"Точность на тестовой выборке: {test_accuracy:.4f}")
        print("\nДетальный отчет по классификации:")
        print(classification_report(test_labels, test_predictions_labels))
        
        return test_accuracy

    def save_model(self, model_dir="models"):
        super().save_model(model_dir)
        joblib.dump(self.kmeans, os.path.join(model_dir, 'kmeans_model.pkl'))
        joblib.dump(self.svm, os.path.join(model_dir, 'svm_model.pkl'))
        joblib.dump(self.scaler, os.path.join(model_dir, 'scaler.pkl'))

    def load_model(self, model_dir="models"):
        super().load_model(model_dir)
        self.kmeans = joblib.load(os.path.join(model_dir, 'kmeans_model.pkl'))
        self.svm = joblib.load(os.path.join(model_dir, 'svm_model.pkl'))
        self.scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
        self.n_clusters = self.kmeans.n_clusters
        return True

    def visualize_sift(self, image_path, output_path=None):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Ошибка загрузки изображения: {image_path}")
            return None
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        
        image_with_keypoints = cv2.drawKeypoints(
            image, keypoints, None, 
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        
        print(f"Найдено {len(keypoints)} ключевых точек на изображении {image_path}")
        
        if output_path:
            cv2.imwrite(output_path, image_with_keypoints)
            print(f"Изображение с SIFT дескрипторами сохранено в: {output_path}")
        
        return image_with_keypoints
