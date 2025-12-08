import os
import argparse
import json
import joblib
from typing import List, Tuple, Dict

import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def infer_label_from_path(rel_path: str) -> str:
    """
    Определяем класс по имени папки во втором уровне пути.
    Примеры:
        NNSUDataset/01_NizhnyNovgorodKremlin/...
        ExtDataset/04_ArkhangelskCathedral/...
        ExtDataset/08_PalaceOfLabor/...
    """
    path = rel_path.replace("\\", "/")
    parts = path.split("/")
    # ожидаем: [NNSUDataset, 01_..., filename]
    if len(parts) >= 2:
        class_dir = parts[1]
    else:
        class_dir = parts[0]

    # Можно вернуть прямо имя папки как метку
    return class_dir


def read_split_file(split_path: str, data_root: str) -> Tuple[List[str], List[str]]:
    """
    Чтение файла разбиения (train.txt или test.txt).
    Формат:
        <relative_path> [optional_label]
    Если метка не указана, берётся из имени папки.
    """
    image_paths = []
    labels = []

    with open(split_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            rel_path = parts[0]

            if len(parts) > 1:
                label = parts[1]
            else:
                label = infer_label_from_path(rel_path)

            full_path = os.path.join(data_root, rel_path)
            image_paths.append(full_path)
            labels.append(label)

    return image_paths, labels


def create_feature_extractor(detector_type: str = "ORB"):
    """
    Создаём детектор/дескриптор ключевых точек.
    Доступные типы: ORB, SIFT (если есть в вашей сборке OpenCV).
    """
    detector_type = detector_type.upper()
    if detector_type == "SIFT":
        if hasattr(cv2, "SIFT_create"):
            detector = cv2.SIFT_create()
            descriptor_dim = 128
        else:
            print("[WARN] SIFT недоступен, переключаюсь на ORB")
            detector = cv2.ORB_create(nfeatures=1000)
            descriptor_dim = 32
    else:
        detector = cv2.ORB_create(nfeatures=1000)
        descriptor_dim = 32

    return detector, descriptor_dim


def extract_descriptors(image_paths: List[str],
                        detector,
                        max_images: int = None) -> Tuple[List[np.ndarray], List[str]]:
    """
    Извлекаем дескрипторы ORB/SIFT для каждого изображения.
    Возвращает:
        - список массивов дескрипторов (по изображению)
        - список путей, для которых дескрипторы реально удалось извлечь
    """
    all_image_descs = []
    valid_paths = []

    if max_images is not None:
        image_paths = image_paths[:max_images]

    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[WARN] Не удалось загрузить изображение: {path}")
            continue
        keypoints, descs = detector.detectAndCompute(img, None)
        if descs is None:
            print(f"[WARN] Нет ключевых точек: {path}")
            continue
        all_image_descs.append(descs)
        valid_paths.append(path)

    return all_image_descs, valid_paths


def build_vocabulary(all_image_descs: List[np.ndarray],
                     dictionary_size: int = 200,
                     batch_size: int = 1000,
                     random_state: int = 42) -> MiniBatchKMeans:
    """
    Обучаем кластеризатор (словарь визуальных слов) по всем дескрипторам.
    """
    # Объединяем все дескрипторы в один массив
    all_descs = np.vstack(all_image_descs).astype(np.float32)
    print(f"[INFO] Общее число дескрипторов: {all_descs.shape[0]}")

    kmeans = MiniBatchKMeans(
        n_clusters=dictionary_size,
        batch_size=batch_size,
        random_state=random_state,
        verbose=1
    )
    kmeans.fit(all_descs)
    return kmeans


def compute_bow_histograms(image_descs: List[np.ndarray],
                           kmeans: MiniBatchKMeans,
                           dictionary_size: int) -> np.ndarray:
    """
    Для каждого изображения строим гистограмму распределения дескрипторов по кластерам.
    """
    X = []

    for descs in image_descs:
        if descs is None or len(descs) == 0:
            hist = np.zeros(dictionary_size, dtype=np.float32)
        else:
            words = kmeans.predict(descs.astype(np.float32))
            hist, _ = np.histogram(words, bins=np.arange(dictionary_size + 1))
            hist = hist.astype(np.float32)
            # L1-нормировка
            hist_sum = hist.sum()
            if hist_sum > 0:
                hist /= hist_sum
        X.append(hist)

    X = np.vstack(X)
    return X


class BowImageClassifier:
    def __init__(self,
                 detector_type: str = "ORB",
                 dictionary_size: int = 200,
                 model_path: str = "bow_model.joblib"):
        self.detector_type = detector_type
        self.dictionary_size = dictionary_size
        self.model_path = model_path

        self.detector = None
        self.descriptor_dim = None
        self.kmeans = None
        self.clf = None
        self.label_encoder = None

    def fit(self,
            train_paths: List[str],
            train_labels: List[str]):
        """
        Обучение модели:
        1. Извлекаем дескрипторы
        2. Строим словарь визуальных слов
        3. Строим BoW-признаки
        4. Обучаем SVM
        """
        print("[INFO] Создаём детектор/дескриптор...")
        self.detector, self.descriptor_dim = create_feature_extractor(self.detector_type)

        print("[INFO] Извлекаем дескрипторы для train...")
        train_descs, valid_train_paths = extract_descriptors(train_paths, self.detector)

        # Синхронизируем метки с реально использованными путями
        path_to_label = {p: y for p, y in zip(train_paths, train_labels)}
        valid_train_labels = [path_to_label[p] for p in valid_train_paths]

        print("[INFO] Обучаем словарь визуальных слов...")
        self.kmeans = build_vocabulary(train_descs, dictionary_size=self.dictionary_size)

        print("[INFO] Строим BoW-признаки для train...")
        X_train = compute_bow_histograms(train_descs, self.kmeans, self.dictionary_size)

        print("[INFO] Кодируем метки классов...")
        self.label_encoder = LabelEncoder()
        y_train = self.label_encoder.fit_transform(valid_train_labels)

        print("[INFO] Обучаем SVM-классификатор...")
        self.clf = LinearSVC(random_state=42)
        self.clf.fit(X_train, y_train)

        print("[INFO] Обучение завершено.")

    def predict(self, image_paths: List[str]) -> np.ndarray:
        """
        Предсказание классов для списка изображений.
        """
        if any(v is None for v in [self.detector, self.kmeans, self.clf, self.label_encoder]):
            raise RuntimeError("Модель не обучена или не загружена.")

        descs, valid_paths = extract_descriptors(image_paths, self.detector)
        X = compute_bow_histograms(descs, self.kmeans, self.dictionary_size)
        y_pred = self.clf.predict(X)
        labels_pred = self.label_encoder.inverse_transform(y_pred)

        # Важно: не все изображения могут быть использованы (если не удалось извлечь дескрипторы),
        # поэтому возвращаем только для valid_paths.
        return labels_pred, valid_paths

    def save(self):
        """
        Сохранение модели на диск.
        """
        state = {
            "detector_type": self.detector_type,
            "dictionary_size": self.dictionary_size,
            "kmeans": self.kmeans,
            "clf": self.clf,
            "label_encoder": self.label_encoder
        }
        joblib.dump(state, self.model_path)
        print(f"[INFO] Модель сохранена в {self.model_path}")

    def load(self):
        """
        Загрузка модели с диска.
        """
        state = joblib.load(self.model_path)
        self.detector_type = state["detector_type"]
        self.dictionary_size = state["dictionary_size"]
        self.kmeans = state["kmeans"]
        self.clf = state["clf"]
        self.label_encoder = state["label_encoder"]

        # Восстанавливаем детектор
        self.detector, self.descriptor_dim = create_feature_extractor(self.detector_type)
        print(f"[INFO] Модель загружена из {self.model_path}")


def evaluate_model(model: BowImageClassifier,
                   test_paths: List[str],
                   test_labels: List[str]):
    """
    Оценка модели на тестовой выборке.
    Выводит accuracy, отчёт по классам, confusion matrix.
    """
    print("[INFO] Оцениваем модель на тестовой выборке...")
    path_to_label = {p: y for p, y in zip(test_paths, test_labels)}

    y_pred_labels, valid_paths = model.predict(test_paths)
    y_true_labels = [path_to_label[p] for p in valid_paths]

    le = LabelEncoder()
    y_true = le.fit_transform(y_true_labels)
    y_pred = le.transform(y_pred_labels)

    acc = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy: {acc:.4f}")

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=le.classes_))

    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))

    return acc


def parse_args():
    parser = argparse.ArgumentParser(
        description="Практическая работа №3. Классификация изображений (мешок слов, OpenCV)"
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Путь до директории с данными (корень, где лежат NNSUDataset и ExtDataset)")
    parser.add_argument("--train_split", type=str, required=True,
                        help="Путь до файла train.txt")
    parser.add_argument("--test_split", type=str, required=True,
                        help="Путь до файла test.txt")
    parser.add_argument("--mode", type=str, choices=["train", "test", "train_test"],
                        default="train_test", help="Режим работы: обучение / тестирование / оба")
    parser.add_argument("--algo", type=str, choices=["bow", "cnn"], default="bow",
                        help="Алгоритм: мешок слов (bow) или нейросеть (cnn). Сейчас реализован bow.")
    parser.add_argument("--detector", type=str, choices=["ORB", "SIFT"], default="ORB",
                        help="Тип детектора/дескриптора для мешка слов")
    parser.add_argument("--dict_size", type=int, default=200,
                        help="Размер словаря визуальных слов (число кластеров)")
    parser.add_argument("--model_path", type=str, default="bow_model.joblib",
                        help="Путь для сохранения/загрузки модели")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.algo != "bow":
        raise NotImplementedError("В данном шаблоне реализован только алгоритм 'мешок слов' (bow).")

    print("[INFO] Загружаем train/test списки...")
    train_paths, train_labels = read_split_file(args.train_split, args.data_dir)
    test_paths, test_labels = read_split_file(args.test_split, args.data_dir)

    model = BowImageClassifier(
        detector_type=args.detector,
        dictionary_size=args.dict_size,
        model_path=args.model_path
    )

    # Обучение
    if args.mode in ("train", "train_test"):
        model.fit(train_paths, train_labels)
        model.save()

        # Для удобства – сохраним mapping меток в JSON
        label_counts: Dict[str, int] = {}
        for lab in train_labels:
            label_counts[lab] = label_counts.get(lab, 0) + 1
        with open("class_distribution.json", "w", encoding="utf-8") as f:
            json.dump(label_counts, f, ensure_ascii=False, indent=4)
        print("[INFO] Распределение классов сохранено в class_distribution.json")

    # Тестирование
    if args.mode in ("test", "train_test"):
        if args.mode == "test":
            model.load()
        acc = evaluate_model(model, test_paths, test_labels)
        print(f"[RESULT] Accuracy на тесте: {acc:.4f}")


if __name__ == "__main__":
    main()
