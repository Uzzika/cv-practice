import os
import argparse
import json
from typing import List, Tuple, Dict

import joblib
import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfTransformer

# ====== Дополнительно для CNN ======
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

# ---------------------------------------------------------------------
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ ЧТЕНИЯ РАЗБИЕНИЯ И МЕТКИ КЛАССА
# ---------------------------------------------------------------------

def infer_label_from_path(rel_path: str) -> str:
    """
    Определяем класс по имени папки во втором уровне пути.
    Примеры:
        ExtDataset/01_NizhnyNovgorodKremlin/...
        ExtDataset/04_ArkhangelskCathedral/...
        ExtDataset/08_PalaceOfLabor/...
    """
    path = rel_path.replace("\\", "/")
    parts = path.split("/")
    if len(parts) >= 2:
        class_dir = parts[1]
    else:
        class_dir = parts[0]
    return class_dir


def read_split_file(split_file, data_root):
    samples = []

    with open(split_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 0:
                continue

            rel_path = parts[0].strip()

            # Универсальная нормализация для Windows/Mac/Linux
            rel_path = rel_path.replace("\\", "/")

            full_path = os.path.join(data_root, rel_path)
            full_path = os.path.normpath(full_path)

            if len(parts) > 1:
                label = parts[1]
            else:
                # если метки нет в файле — извлекаем из имени папки
                label = os.path.basename(os.path.dirname(rel_path))

            samples.append((full_path, label))

    return samples

# ---------------------------------------------------------------------
# ВИЗУАЛИЗАЦИЯ ЭТАПОВ «МЕШКА СЛОВ»
# ---------------------------------------------------------------------

def visualize_keypoints(image_path: str, detector, out_path: str):
    """
    Визуализация ключевых точек (ORB/SIFT) на одном изображении.
    Результат сохраняется в out_path.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"[WARN] Не удалось загрузить изображение для визуализации: {image_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kps, _ = detector.detectAndCompute(gray, None)
    if not kps:
        print(f"[WARN] Не удалось найти ключевые точки для визуализации: {image_path}")
        return

    vis = cv2.drawKeypoints(img, kps, None,
                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite(out_path, vis)
    print(f"[INFO] Визуализация ключевых точек сохранена в {out_path}")


def visualize_bow_histogram(hist: np.ndarray, out_path: str):
    """
    Визуализация гистограммы BoW-признаков (распределение по визуальным словам).
    Если matplotlib не установлен, выводим предупреждение и пропускаем шаг.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib не установлен, пропускаю визуализацию гистограммы BoW.")
        return

    plt.figure(figsize=(8, 3))
    plt.bar(np.arange(len(hist)), hist)
    plt.title("BoW-гистограмма для одного изображения")
    plt.xlabel("Индекс визуального слова")
    plt.ylabel("Нормированная частота")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[INFO] Визуализация BoW-гистограммы сохранена в {out_path}")

# ---------------------------------------------------------------------
# ЧАСТЬ 1. МЕШОК СЛОВ (BOW)
# ---------------------------------------------------------------------

def create_feature_extractor(detector_type: str = "ORB"):
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
        if descs is not None:
            descs = descs.astype(np.float32)
            descs /= (descs.sum(axis=1, keepdims=True) + 1e-7)
            descs = np.sqrt(descs)
        all_image_descs.append(descs)
        valid_paths.append(path)

    return all_image_descs, valid_paths


def build_vocabulary(all_image_descs: List[np.ndarray],
                     dictionary_size: int = 400,
                     batch_size: int = 1000,
                     random_state: int = 42) -> MiniBatchKMeans:
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
    X = []
    for descs in image_descs:
        if descs is None or len(descs) == 0:
            hist = np.zeros(dictionary_size, dtype=np.float32)
        else:
            words = kmeans.predict(descs.astype(np.float32))
            hist, _ = np.histogram(words, bins=np.arange(dictionary_size + 1))
            hist = hist.astype(np.float32)
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
        self.tfidf = None

    def fit(self,
            train_paths: List[str],
            train_labels: List[str]):
        print("[INFO] Создаём детектор/дескриптор...")
        self.detector, self.descriptor_dim = create_feature_extractor(self.detector_type)

        print("[INFO] Извлекаем дескрипторы для train...")
        train_descs, valid_train_paths = extract_descriptors(train_paths, self.detector)

        path_to_label = {p: y for p, y in zip(train_paths, train_labels)}
        valid_train_labels = [path_to_label[p] for p in valid_train_paths]

        print("[INFO] Обучаем словарь визуальных слов...")
        self.kmeans = build_vocabulary(train_descs, dictionary_size=self.dictionary_size)

        print("[INFO] Строим BoW-признаки для train...")
        X_train = compute_bow_histograms(train_descs, self.kmeans, self.dictionary_size)

        self.tfidf = TfidfTransformer()
        X_train = self.tfidf.fit_transform(X_train).toarray()

        # === ВИЗУАЛИЗАЦИЯ ЭТАПОВ BOW ===
        if len(valid_train_paths) > 0:
            sample_idx = 0  # первый успешно обработанный пример
            sample_path = valid_train_paths[sample_idx]
            sample_label = valid_train_labels[sample_idx]
            sample_hist = X_train[sample_idx]

            os.makedirs("visualizations", exist_ok=True)

            kp_out = os.path.join(
                "visualizations",
                f"keypoints_{sample_label}.jpg"
            )
            hist_out = os.path.join(
                "visualizations",
                f"bow_hist_{sample_label}.png"
            )

            print("[INFO] Визуализируем ключевые точки и BoW-гистограмму для одного изображения...")
            visualize_keypoints(sample_path, self.detector, kp_out)
            visualize_bow_histogram(sample_hist, hist_out)
        else:
            print("[WARN] Нет ни одного изображения с дескрипторами для визуализации.")

        print("[INFO] Кодируем метки классов...")
        self.label_encoder = LabelEncoder()
        y_train = self.label_encoder.fit_transform(valid_train_labels)

        print("[INFO] Обучаем SVM-классификатор...")
        self.clf = LinearSVC(
            C=5,
            class_weight="balanced",
            max_iter=10000
        )
        self.clf.fit(X_train, y_train)

        print("[INFO] Обучение (BOW) завершено.")

    def predict(self, image_paths: List[str]):
        if any(v is None for v in [self.detector, self.kmeans, self.clf, self.label_encoder]):
            raise RuntimeError("Модель не обучена или не загружена.")

        descs, valid_paths = extract_descriptors(image_paths, self.detector)
        if len(descs) == 0:
            raise RuntimeError("Не удалось извлечь дескрипторы ни для одного изображения в тесте.")

        X = compute_bow_histograms(descs, self.kmeans, self.dictionary_size)
        X = self.tfidf.transform(X).toarray()
        y_pred = self.clf.predict(X)
        labels_pred = self.label_encoder.inverse_transform(y_pred)
        return labels_pred, valid_paths

    def save(self):
        state = {
            "detector_type": self.detector_type,
            "dictionary_size": self.dictionary_size,
            "kmeans": self.kmeans,
            "clf": self.clf,
            "label_encoder": self.label_encoder
        }
        joblib.dump(state, self.model_path)
        print(f"[INFO] Модель (BOW) сохранена в {self.model_path}")

    def load(self):
        state = joblib.load(self.model_path)
        self.detector_type = state["detector_type"]
        self.dictionary_size = state["dictionary_size"]
        self.kmeans = state["kmeans"]
        self.clf = state["clf"]
        self.label_encoder = state["label_encoder"]

        self.detector, self.descriptor_dim = create_feature_extractor(self.detector_type)
        print(f"[INFO] Модель (BOW) загружена из {self.model_path}")


def evaluate_bow_model(model: BowImageClassifier,
                       test_paths: List[str],
                       test_labels: List[str]):
    print("[INFO] Оцениваем BOW-модель на тестовой выборке...")
    path_to_label = {p: y for p, y in zip(test_paths, test_labels)}

    y_pred_labels, valid_paths = model.predict(test_paths)
    y_true_labels = [path_to_label[p] for p in valid_paths]

    le = model.label_encoder
    y_true = le.transform(y_true_labels)
    y_pred = le.transform(y_pred_labels)

    acc = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy: {acc:.4f}\n")

    print(classification_report(y_true, y_pred, target_names=le.classes_, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))

    return acc

# ---------------------------------------------------------------------
# ЧАСТЬ 2. НЕЙРОСЕТЕВОЙ КЛАССИФИКАТОР (TRANSFER LEARNING RESNET18)
# ---------------------------------------------------------------------

class SplitFileImageDataset(Dataset):
    """
    PyTorch Dataset, который читает пути и метки из наших списков.
    """
    def __init__(self, image_paths: List[str], labels: List[str], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

        self.classes = sorted(list(set(labels)))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.targets = [self.class_to_idx[l] for l in labels]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = self.targets[idx]

        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        return img, label


def create_dataloaders(train_paths, train_labels,
                       test_paths, test_labels,
                       batch_size=16):
    """
    Создаём DataLoader'ы для обучения и теста.
    Используем стандартные трансформации для ResNet18.
    """
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = SplitFileImageDataset(train_paths, train_labels, transform=train_transform)
    test_dataset = SplitFileImageDataset(test_paths, test_labels, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader, train_dataset.classes


def create_cnn_model(num_classes: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Используем устройство: {device}")

    try:
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    except Exception:
        model = models.resnet18(pretrained=True)

    # Замораживаем всё
    for param in model.parameters():
        param.requires_grad = False

    # 🔥 Размораживаем последний residual-блок
    for param in model.layer4.parameters():
        param.requires_grad = True

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    model = model.to(device)
    return model, device


def train_cnn(model, device, train_loader,
              epochs=5, lr=1e-4):
    """
    Обучение последнего слоя ResNet18.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )

    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total if total > 0 else 0
        epoch_acc = correct / total if total > 0 else 0
        print(f"[CNN] Epoch {epoch}/{epochs} - loss: {epoch_loss:.4f}, acc: {epoch_acc:.4f}")


def evaluate_cnn(model, device, data_loader, class_names):
    """
    Оценка CNN на тестовой выборке.
    Возвращает accuracy и печатает отчёт.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.inference_mode():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = outputs.max(1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    print(f"\n[CNN] Accuracy: {acc:.4f}\n")

    print(classification_report(all_labels, all_preds,
                                target_names=class_names,
                                zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(all_labels, all_preds))

    return acc


def save_cnn_model(model, class_names, path: str):
    state = {
        "model_state": model.state_dict(),
        "class_names": class_names,
    }
    torch.save(state, path)
    print(f"[INFO] CNN-модель сохранена в {path}")


def load_cnn_model(path: str):
    state = torch.load(path, map_location="cpu")
    class_names = state["class_names"]
    num_classes = len(class_names)
    model, device = create_cnn_model(num_classes)
    model.load_state_dict(state["model_state"])
    model = model.to(device)
    print(f"[INFO] CNN-модель загружена из {path}")
    return model, device, class_names

# ---------------------------------------------------------------------
# ПАРСЕР АРГУМЕНТОВ И main()
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Практическая работа №3. Классификация изображений (мешок слов / CNN, OpenCV)"
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
                        help="Алгоритм: мешок слов (bow) или нейросеть (cnn).")
    # Параметры BOW
    parser.add_argument("--detector", type=str, choices=["ORB", "SIFT"], default="ORB",
                        help="Тип детектора/дескриптора для мешка слов")
    parser.add_argument("--dict_size", type=int, default=1000,
                        help="Размер словаря визуальных слов (число кластеров)")
    # Общий путь к модели
    parser.add_argument("--model_path", type=str, default="model.bin",
                        help="Путь для сохранения/загрузки модели (bow_model.joblib или cnn_model.pth)")
    # Параметры CNN
    parser.add_argument("--cnn_epochs", type=int, default=5,
                        help="Количество эпох обучения CNN")
    parser.add_argument("--cnn_batch_size", type=int, default=16,
                        help="Размер батча для CNN")
    parser.add_argument("--cnn_lr", type=float, default=1e-4,
                        help="Learning rate для CNN")

    return parser.parse_args()


def main():
    args = parse_args()

    print("[INFO] Загружаем train/test списки...")
    train_samples = read_split_file(args.train_split, args.data_dir)
    test_samples = read_split_file(args.test_split, args.data_dir)

    train_paths = [s[0] for s in train_samples]
    train_labels = [s[1] for s in train_samples]

    test_paths = [s[0] for s in test_samples]
    test_labels = [s[1] for s in test_samples]

    # -------------------- BOW --------------------
    if args.algo == "bow":
        model = BowImageClassifier(
            detector_type=args.detector,
            dictionary_size=args.dict_size,
            model_path=args.model_path if args.model_path != "model.bin" else "bow_model.joblib"
        )

        if args.mode in ("train", "train_test"):
            model.fit(train_paths, train_labels)
            model.save()

            label_counts: Dict[str, int] = {}
            for lab in train_labels:
                label_counts[lab] = label_counts.get(lab, 0) + 1
            with open("class_distribution.json", "w", encoding="utf-8") as f:
                json.dump(label_counts, f, ensure_ascii=False, indent=4)
            print("[INFO] Распределение классов сохранено в class_distribution.json")

        if args.mode in ("test", "train_test"):
            if args.mode == "test":
                model.load()
            acc = evaluate_bow_model(model, test_paths, test_labels)
            print(f"[RESULT] BOW accuracy на тесте: {acc:.4f}")

    # -------------------- CNN --------------------
    elif args.algo == "cnn":
        cnn_model_path = args.model_path if args.model_path != "model.bin" else "cnn_model.pth"

        if args.mode in ("train", "train_test"):
            train_loader, test_loader, class_names = create_dataloaders(
                train_paths, train_labels,
                test_paths, test_labels,
                batch_size=args.cnn_batch_size
            )
            model, device = create_cnn_model(num_classes=len(class_names))
            train_cnn(model, device, train_loader,
                      epochs=args.cnn_epochs, lr=args.cnn_lr)
            save_cnn_model(model, class_names, cnn_model_path)

        if args.mode in ("test", "train_test"):
            if args.mode == "test":
                model, device, class_names = load_cnn_model(cnn_model_path)
                _, test_loader, _ = create_dataloaders(
                    train_paths, train_labels,
                    test_paths, test_labels,
                    batch_size=args.cnn_batch_size
                )
            else:
                model, device, class_names = load_cnn_model(cnn_model_path)
                _, test_loader, _ = create_dataloaders(
                    train_paths, train_labels,
                    test_paths, test_labels,
                    batch_size=args.cnn_batch_size
                )

            acc = evaluate_cnn(model, device, test_loader, class_names)
            print(f"[RESULT] CNN accuracy на тесте: {acc:.4f}")


if __name__ == "__main__":
    main()
