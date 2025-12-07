import os
import cv2
import numpy as np
from abc import ABC, abstractmethod
from sklearn.metrics import classification_report

class BaseClassifier(ABC):
    def __init__(self, model_dir='bow_model'):
        """
        Инициализация базового классификатора

        Args:
            model_dir (str): Директория для сохранения/загрузки моделей
        """
        self.model_dir = model_dir
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

    def detect_label_from_path(self, image_path):
        """
        Определение метки класса из пути к изображению

        Args:
            image_path (str): Путь к изображению

        Returns:
            str: Метка класса
        """
        # Предполагается, что изображения находятся в подпапках с названиями классов
        return os.path.basename(os.path.dirname(image_path))

    def load_data(self, data_file, data_dir):
        """
        Загрузка данных из файла со списком изображений

        Args:
            data_file (str): Путь к файлу со списком изображений
            data_dir (str): Корневая директория с изображениями

        Returns:
            tuple: Списки путей к изображениям и их меток
        """
        image_paths = []
        labels = []

        with open(data_file, 'r') as f:
            for line in f:
                relative_path = line.strip()
                if relative_path:
                    full_path = os.path.join(data_dir, relative_path)
                    if os.path.exists(full_path):
                        image_paths.append(full_path)
                        labels.append(self.detect_label_from_path(full_path))

        return image_paths, labels

    def load_image(self, image_path):
        """
        Загрузка и предобработка изображения

        Args:
            image_path (str): Путь к изображению

        Returns:
            numpy.ndarray: Загруженное изображение
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")
        return image

    def evaluate(self, y_true, y_pred, target_names):
        """
        Оценка качества классификации

        Args:
            y_true (list): Истинные метки
            y_pred (list): Предсказанные метки
            target_names (list): Названия классов

        Returns:
            str: Отчет с метриками качества
        """
        return classification_report(y_true, y_pred, target_names=target_names)

    @abstractmethod
    def train(self, train_paths, train_labels):
        """
        Обучение классификатора

        Args:
            train_paths (list): Пути к обучающим изображениям
            train_labels (list): Метки обучающих изображений
        """
        pass

    @abstractmethod
    def test(self, test_paths, test_labels=None):
        """
        Тестирование классификатора

        Args:
            test_paths (list): Пути к тестовым изображениям
            test_labels (list, optional): Метки тестовых изображений

        Returns:
            tuple: Предсказанные метки и точность (если доступны истинные метки)
        """
        pass

