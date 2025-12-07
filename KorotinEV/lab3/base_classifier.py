import os
import json
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

class BaseClassifier:
    def __init__(self, algorithm, image_size=(224, 224), class_names=None):
        self.algorithm = algorithm
        self.image_size = image_size
        
        if class_names is None:
            self.class_names = ['Архангельский собор', 'Дворец труда', 'Нижегородский Кремль']
        else:
            self.class_names = class_names
            
        self.label_to_id = {name: i for i, name in enumerate(self.class_names)}
        
    def detect_label_from_path(self, file_path):
        if '01_NizhnyNovgorodKremlin' in file_path:
            return 'Нижегородский Кремль'
        elif '08_PalaceOfLabor' in file_path:
            return 'Дворец труда'
        elif '04_ArkhangelskCathedral' in file_path:
            return 'Архангельский собор'
        else:
            print(f"Не удалось определить метку для файла: {file_path}")
            return None

    def load_data(self, file_list_path, images_dir="."):
        with open(file_list_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        image_paths = []
        labels = []
        label_ids = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            full_path = os.path.join(images_dir, line)
            
            if os.path.exists(full_path):
                label = self.detect_label_from_path(full_path)
                if label is not None:
                    image_paths.append(full_path)
                    labels.append(label)
                    label_ids.append(self.label_to_id[label])
                else:
                    print(f"Пропущен файл (не определена метка): {full_path}")
            else:
                print(f"Файл не найден: {full_path}")
        
        print(f"Загружено {len(image_paths)} изображений")
        print(f"Распределение по классам:")
        for label_name in self.class_names:
            count = labels.count(label_name)
            print(f"  {label_name}: {count} изображений")
        
        return image_paths, labels, label_ids

    def save_model(self, model_dir="models"):
        os.makedirs(model_dir, exist_ok=True)
        
        metadata = {
            'algorithm': self.algorithm,
            'image_size': self.image_size,
            'class_names': self.class_names
        }
        
        with open(os.path.join(model_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"Модель сохранена в директории: {model_dir}")

    def load_model(self, model_dir="models"):
        with open(os.path.join(model_dir, 'metadata.json'), 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        self.algorithm = metadata['algorithm']
        self.class_names = metadata['class_names']
        self.image_size = tuple(metadata['image_size'])
        self.label_to_id = {name: i for i, name in enumerate(self.class_names)}
        
        print(f"Модель загружена из директории: {model_dir}")
        print(f"Алгоритм: {self.algorithm}")
        return True

    def train(self, train_file, images_dir="."):
        raise NotImplementedError("Метод train должен быть реализован в дочернем классе")

    def test(self, test_file, images_dir="."):
        raise NotImplementedError("Метод test должен быть реализован в дочернем классе")
