import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

import os
from collections import defaultdict

from classifier import Classifier


class BoWClassifier(Classifier):
    def __init__(self, args):
        super().__init__(args)

        self.kmeans = None
        self.scaler = None
        self.clf = None
        self.label_encoder = None

    def _extract_descriptors(self, image_paths, max_kp=1000):
        det = cv2.SIFT_create(nfeatures=max_kp)

        descs = []
        for p in image_paths:
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Не удалось открыть изображение: {p}")
            _, d = det.detectAndCompute(img, None)
            if d is None:
                d = np.zeros((0, 128), dtype=np.float32)
            else:
                d = d.astype(np.float32)
            descs.append(d)
        return descs

    def _build_vocab(self, train_descs, k):
        pool = [d for d in train_descs if d is not None and len(d) > 0]
        if not pool:
            raise RuntimeError("Нет дескрипторов для обучения словаря.")
        X = np.vstack(pool).astype(np.float32)
        kmeans = KMeans(n_clusters=k, n_init=10)
        kmeans.fit(X)
        return kmeans

    def _bow_hists(self, descs, kmeans):
        k = kmeans.n_clusters
        H = np.zeros((len(descs), k), dtype=np.float32)
        for i, d in enumerate(descs):
            if d is None or len(d) == 0:
                continue
            words = kmeans.predict(d)
            np.add.at(H[i], words, 1.0)
            n = np.linalg.norm(H[i])
            if n > 0:
                H[i] /= n
        return H

    def _draw_sift_keypoints(self, img_bgr, max_kp=1000):
        sift = cv2.SIFT_create(nfeatures=max_kp)

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        kps = sift.detect(gray, None)

        kps = sorted(kps, key=lambda kp: kp.response, reverse=True)[:max_kp]
        vis = cv2.drawKeypoints(
            img_bgr, kps, None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        return vis, len(kps)

    def _save_keypoints_for_items(self, items, out_dir, per_class=2, max_kp=1000):
        os.makedirs(out_dir, exist_ok=True)
        by_class = defaultdict(list)
        for p, lbl in items:
            by_class[lbl].append(p)

        saved = []
        for lbl, paths in by_class.items():
            for src_path in paths[:per_class]:
                img = cv2.imread(src_path, cv2.IMREAD_COLOR)
                if img is None:
                    print(f"Не удалось открыть: {src_path}")
                    continue

                base = os.path.splitext(os.path.basename(src_path))[0]
                vis, n_kp = self._draw_sift_keypoints(img, max_kp=max_kp)
                out_path = os.path.join(out_dir, f"{lbl}_{base}_kps.jpg")
                ok = cv2.imwrite(out_path, vis)
                if ok:
                    saved.append(out_path)
                    print(f"{lbl}: {src_path} -> {out_path} (kps={n_kp})")
                else:
                    print(f"Не удалось сохранить: {out_path}")
        return saved

    def _save_model(self, path):
        if any(x is None for x in (self.kmeans, self.scaler, self.clf, self.label_encoder)):
            raise RuntimeError(
                "Нельзя сохранить: модель не полностью инициализирована")
        model = {
            "kmeans": self.kmeans,
            "scaler": self.scaler,
            "clf": self.clf,
            "label_encoder": self.label_encoder,
            "params": {
                "k": self.args.k,
                "C": self.args.C,
                "max_kp": self.args.max_kp,
            },
        }
        with open(path, "wb") as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_model(self, path):
        with open(path, "rb") as f:
            model = pickle.load(f)
        self.kmeans = model["kmeans"]
        self.scaler = model["scaler"]
        self.clf = model["clf"]
        self.label_encoder = model["label_encoder"]

        params = model.get("params", {})
        if "k" in params:
            self.args.k = params["k"]
        if "C" in params:
            self.args.C = params["C"]
        if "max_kp" in params:
            self.args.max_kp = params["max_kp"]

    def _ensure_model_loaded(self):
        if self.kmeans is not None and self.scaler is not None and self.clf is not None:
            return
        if not getattr(self.args, "model_in", None):
            raise ValueError(
                "Не задан --model_in"
            )
        print(f"Загружаем модель из {self.args.model_in}")
        self._load_model(self.args.model_in)

    def train(self, train_items):
        k = getattr(self.args, "k", 300)
        C = getattr(self.args, "C", 1.0)
        max_kp = getattr(self.args, "max_kp", 1000)

        viz_dir = getattr(self.args, "vizualize_dir", None)
        if viz_dir:
            per_class = getattr(self.args, "viz_per_class", 2)
            self._save_keypoints_for_items(
                train_items,
                out_dir=viz_dir,
                per_class=per_class,
                max_kp=max_kp,
            )

        paths = [p for p, _ in train_items]
        labels = [c for _, c in train_items]

        descs = self._extract_descriptors(paths, max_kp)
        kmeans = self. _build_vocab(descs, k=k)

        X = self._bow_hists(descs, kmeans)

        scaler = StandardScaler(with_mean=False)
        Xs = scaler.fit_transform(X)

        le = LabelEncoder().fit(labels)
        y = le.transform(labels)

        clf = LinearSVC(C=C, class_weight="balanced", max_iter=5000, dual=True)
        clf.fit(Xs, y)

        self.kmeans = kmeans
        self.scaler = scaler
        self.clf = clf
        self.label_encoder = le

        print("Модель BoW обучена")

        if getattr(self.args, "model_out", None):
            self._save_model(self.args.model_out)
            print(f"Модель сохранена: {self.args.model_out}")

    def test(self, test_items):
        max_kp = getattr(self.args, "max_kp", 1000)

        self._ensure_model_loaded()

        paths = [p for p, _ in test_items]
        labels = [c for _, c in test_items]

        descs = self._extract_descriptors(paths, max_kp)
        X = self._bow_hists(descs, self.kmeans)
        Xs = self.scaler.transform(X)

        y_true = self.label_encoder.transform(labels)
        y_pred = self.clf.predict(Xs)
        classes = list(self.label_encoder.classes_)

        report = {
            "accuracy": accuracy_score(y_true, y_pred),
            "classification_report": classification_report(y_true, y_pred, target_names=classes, digits=4),
            "confusion_matrix": confusion_matrix(y_true, y_pred)
        }

        print(f"Accuracy: {report['accuracy']:.4f}")
        print("Classification report:\n", report["classification_report"])
        print("Confusion matrix:\n", report["confusion_matrix"])

        self.last_report = (y_true, y_pred, classes, report)
