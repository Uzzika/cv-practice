import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import SVC
import joblib
import os
from .base_classifier import BaseClassifier

class BOWClassifier(BaseClassifier):
    def __init__(self, n_clusters=100, model_dir='bow_model'):
        """
        Инициализация классификатора на основе метода "мешок слов"

        Args:
            n_clusters (int): Количество кластеров (размер словаря)
            model_dir (str): Директория для сохранения/загрузки моделей
        """
        super().__init__(model_dir)
        self.n_clusters = n_clusters
        self.sift = cv2.SIFT_create()
        self.kmeans = None
        self.classifier = None
        self.class_names = None

    def extract_features(self, image):
        """
        Извлечение SIFT-дескрипторов из изображения

        Args:
            image: Изображение

        Returns:
            numpy.ndarray: Матрица дескрипторов
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        return descriptors

    def build_vocabulary(self, all_descriptors):
        """
        Построение визуального словаря путем кластеризации дескрипторов

        Args:
            all_descriptors (numpy.ndarray): Матрица всех дескрипторов
        """
        print("Построение визуального словаря...")
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.kmeans.fit(all_descriptors)

    def descr_to_histogram(self, descriptors):
        """
        Преобразование дескрипторов в гистограмму визуальных слов

        Args:
            descriptors (numpy.ndarray): Матрица дескрипторов

        Returns:
            numpy.ndarray: Гистограмма визуальных слов
        """
        if descriptors is None:
            return np.zeros(self.n_clusters)

        predictions = self.kmeans.predict(descriptors)
        histogram = np.bincount(predictions, minlength=self.n_clusters)

        # Нормализация гистограммы
        if histogram.sum() > 0:
            histogram = histogram / histogram.sum()

        return histogram

    def train(self, train_paths, train_labels):
        """
        Обучение классификатора

        Args:
            train_paths (list): Пути к обучающим изображениям
            train_labels (list): Метки обучающих изображений
        """
        print("Начало обучения BOW классификатора...")

        # Сохраняем уникальные метки классов
        self.class_names = sorted(list(set(train_labels)))

        # Извлечение дескрипторов из всех изображений
        all_descriptors = []
        for path in train_paths:
            print(".")
            image = self.load_image(path)
            descriptors = self.extract_features(image)
            if descriptors is not None:
                all_descriptors.append(descriptors)

        # Объединение всех дескрипторов
        all_descriptors = np.vstack(all_descriptors)

        # Построение визуального словаря
        self.build_vocabulary(all_descriptors)

        # Формирование обучающей выборки
        train_histograms = []
        for path in train_paths:
            print(".")
            image = self.load_image(path)
            descriptors = self.extract_features(image)
            histogram = self.descr_to_histogram(descriptors)
            train_histograms.append(histogram)

        # Обучение SVM классификатора
        self.classifier = SVC(kernel='rbf', probability=True)
        self.classifier.fit(train_histograms, train_labels)

        # Сохранение модели
        self.save_model()
        print("Обучение завершено")

    def test(self, test_paths, test_labels=None):
        """
        Тестирование классификатора

        Args:
            test_paths (list): Пути к тестовым изображениям
            test_labels (list, optional): Метки тестовых изображений

        Returns:
            tuple: Предсказанные метки и точность (если доступны истинные метки)
        """
        if self.kmeans is None or self.classifier is None:
            self.load_model()

        test_histograms = []
        for path in test_paths:
            image = self.load_image(path)
            descriptors = self.extract_features(image)
            histogram = self.descr_to_histogram(descriptors)
            test_histograms.append(histogram)

        predictions = self.classifier.predict(test_histograms)

        accuracy = None
        if test_labels is not None:
            correct = sum(1 for p, t in zip(predictions, test_labels) if p == t)
            accuracy = correct / len(test_labels)
            print(f"\nТочность классификации: {accuracy:.3f}")
            print("\nОтчет по классификации:")
            print(self.evaluate(test_labels, predictions, self.class_names))

        return predictions, accuracy

    def save_model(self):
        """Сохранение модели в файл"""
        model_data = {
            'kmeans': self.kmeans,
            'classifier': self.classifier,
            'class_names': self.class_names,
            'n_clusters': self.n_clusters
        }
        joblib.dump(model_data, os.path.join(self.model_dir, 'bow_model.joblib'))

    def load_model(self):
        """Загрузка модели из файла"""
        model_path = os.path.join(self.model_dir, 'bow_model.joblib')
        if not os.path.exists(model_path):
            raise ValueError("Модель не найдена")

        model_data = joblib.load(model_path)
        self.kmeans = model_data['kmeans']
        self.classifier = model_data['classifier']
        self.class_names = model_data['class_names']
        self.n_clusters = model_data['n_clusters']

    def show_keypoints(self, image_path, output_path=None, draw_style='rich'):
        """
        Визуализация SIFT-ключевых точек на изображении
        
        Args:
            image_path (str): Путь к изображению
            output_path (str, optional): Путь для сохранения результата
            draw_style (str): Стиль отрисовки ('rich', 'simple', 'not_scaled')
        
        Returns:
            tuple: (изображение с ключевыми точками, количество точек, статистика)
        """
        image = self.load_image(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Извлечение ключевых точек и дескрипторов
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        
        if keypoints is None or len(keypoints) == 0:
            print(f"Предупреждение: не найдено ключевых точек на изображении {image_path}")
            return None, 0, {}
        
        # Выбор стиля отрисовки
        if draw_style == 'rich':
            flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        elif draw_style == 'simple':
            flags = cv2.DRAW_MATCHES_FLAGS_DEFAULT
        elif draw_style == 'not_scaled':
            flags = cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS
        else:
            flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        
        # Отрисовка ключевых точек
        image_with_keypoints = cv2.drawKeypoints(
            image, keypoints, None,
            color=(0, 255, 0),  # Зеленый цвет
            flags=flags
        )
        
        # Вычисление статистики
        sizes = [kp.size for kp in keypoints]
        angles = [kp.angle for kp in keypoints]
        responses = [kp.response for kp in keypoints]
        
        stats = {
            'total_keypoints': len(keypoints),
            'avg_size': np.mean(sizes) if sizes else 0,
            'avg_angle': np.mean(angles) if angles else 0,
            'avg_response': np.mean(responses) if responses else 0,
            'descriptors_shape': descriptors.shape if descriptors is not None else None
        }
        
        # Вывод информации
        print(f"\n=== Анализ SIFT-дескрипторов для {os.path.basename(image_path)} ===")
        print(f"Найдено ключевых точек: {stats['total_keypoints']}")
        print(f"Средний размер области: {stats['avg_size']:.2f} пикселей")
        print(f"Средний угол ориентации: {stats['avg_angle']:.2f} градусов")
        print(f"Средняя сила отклика: {stats['avg_response']:.4f}")
        if stats['descriptors_shape']:
            print(f"Размерность дескрипторов: {stats['descriptors_shape']}")
        
        # Сохранение результата
        if output_path:
            cv2.imwrite(output_path, image_with_keypoints)
            print(f"\nИзображение с ключевыми точками сохранено: {output_path}")
        
        return image_with_keypoints, len(keypoints), stats

