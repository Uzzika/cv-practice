import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import joblib
class BOW:
    def __init__(self,clusters_name = None,clf_name = None, k_nearest = 5, clusters = 500, batch_size=1000):
        self.clusters = clusters
        self.batch_size=batch_size
        self.clusters_name = clusters_name
        self.clf_name = clf_name
        self.k_nearest = k_nearest
        self.descriptors_list = None
        self.kmeans = None
        self.histograms = None
        self.clf = None
        self.scaler = None

    def sift_descr(self,images):
        sift = cv2.SIFT_create()
        self.descriptors_list = []
        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = sift.detectAndCompute(gray, None)
            if descriptors is not None:
                self.descriptors_list.append(descriptors)

    def print_descr(self, img):
        sift = cv2.SIFT_create()
        self.descriptors_list = []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        img_kp = cv2.drawKeypoints(img,keypoints,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow("Дескрипторы", img_kp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def bag_of_words(self):
        all_descriptors = np.vstack(self.descriptors_list)
        if self.clusters_name == "KMeans":
            self.kmeans = KMeans(n_clusters=self.clusters,random_state=42)
        elif self.clusters_name == "MiniBatch":
            self.kmeans = MiniBatchKMeans(n_clusters=self.clusters, random_state=42, batch_size=self.batch_size)
        self.kmeans.fit(all_descriptors)
        
    def bow_histograms(self):
        self.histograms = []
        for descriptors in self.descriptors_list:
            words = self.kmeans.predict(descriptors)
            hist, _ = np.histogram(words, bins=np.arange(self.kmeans.n_clusters + 1))
            self.histograms.append(hist)

    def train_bow_model(self, labels):
        X = self.histograms
        self.scaler = StandardScaler().fit(X)
        X_scaled = self.scaler.transform(X)
        if self.clf_name == "SVC":
            self.clf = SVC(kernel = 'linear', probability = True, random_state = 42)
        elif self.clf_name == "KNN":
            self.clf = clf = KNeighborsClassifier(n_neighbors=self.k_nearest)
        self.clf.fit(X_scaled, labels)

    def test_bow_model(self, labels=None):
        X = self.scaler.transform(self.histograms)
        preds = self.clf.predict(X)
        if labels is not None:
            acc = accuracy_score(labels, preds)
            return preds, acc
        return preds

    def save_model(self, output_path):
        joblib.dump({'clf': self.clf, 'scaler': self.scaler, 'bag': self.kmeans}, output_path)

    def my_load_model(self, model_path):
        data = joblib.load(model_path)
        self.clf = data['clf']
        self.scaler = data['scaler']
        self.kmeans = data['bag']
