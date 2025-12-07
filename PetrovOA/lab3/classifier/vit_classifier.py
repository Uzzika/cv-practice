import cv2
import numpy as np
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from PIL import Image
from tqdm import tqdm
from .base_classifier import BaseClassifier


class ImageDataset(Dataset):
    """Dataset для загрузки изображений"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Загрузка изображения
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        
        if image is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")
        
        # Конвертация BGR -> RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        
        # Применение трансформаций
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        
        return image, label


class ViTClassifier(BaseClassifier):
    """
    Классификатор на основе Vision Transformer (DINOv2)
    
    Использует предобученную модель DINOv2-small с замороженным backbone
    и обучаемой классификационной головой.
    
    Обучение: GPU
    Инференс: CPU
    """
    
    def __init__(self, model_dir='vit_model', image_size=224,
                 learning_rate=0.001, batch_size=16, epochs=20):
        """
        Инициализация ViT классификатора
        
        Args:
            model_dir (str): Директория для сохранения/загрузки моделей
            image_size (int): Размер входного изображения
            learning_rate (float): Скорость обучения
            batch_size (int): Размер батча
            epochs (int): Количество эпох обучения
        """
        super().__init__(model_dir)
        self.image_size = image_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        
        self.model = None
        self.label_encoder = LabelEncoder()
        self.device = None  # Будет установлено при обучении/инференсе
        
        # Трансформации для обучения (с аугментацией)
        self.train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Трансформации для инференса (без аугментации)
        self.test_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def create_model(self, n_classes):
        """
        Создание модели на основе Vision Transformer
        
        Args:
            n_classes (int): Количество классов
        """
        print("\n=== Загрузка предобученной модели Vision Transformer ===")
        print("Используется модель из torchvision (загрузка из PyTorch Hub)")
        
        # Используем ViT из torchvision (более надежный источник)
        try:
            from torchvision.models import vit_b_16, ViT_B_16_Weights
            print("\nЗагрузка ViT-B/16 с весами ImageNet-1K...")
            backbone = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
            
            # Извлекаем encoder (без классификационной головы)
            # ViT состоит из: conv_proj, encoder, heads
            # Нам нужен только encoder
            class ViTEncoder(nn.Module):
                def __init__(self, vit_model):
                    super().__init__()
                    self.conv_proj = vit_model.conv_proj
                    self.encoder = vit_model.encoder
                    self.class_token = vit_model.class_token
                    
                def forward(self, x):
                    # Преобразование изображения в патчи
                    x = self.conv_proj(x)
                    x = x.flatten(2).transpose(1, 2)
                    
                    # Добавление class token
                    batch_size = x.shape[0]
                    class_token = self.class_token.expand(batch_size, -1, -1)
                    x = torch.cat([class_token, x], dim=1)
                    
                    # Encoder
                    x = self.encoder(x)
                    
                    # Берем только class token (первый токен)
                    return x[:, 0]
            
            backbone = ViTEncoder(backbone)
            embedding_dim = 768  # ViT-B/16 имеет 768-мерные эмбеддинги
            print("✓ Загружена модель: ViT-B/16 (torchvision, ImageNet-1K)")
            
        except Exception as e:
            print(f"⚠ Не удалось загрузить ViT-B/16: {e}")
            print("  Попытка использовать ResNet50 как альтернативу...")
            
            # Альтернатива: ResNet50
            from torchvision.models import resnet50, ResNet50_Weights
            backbone_full = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            # Убираем последний FC слой
            backbone = nn.Sequential(*list(backbone_full.children())[:-1])
            # Добавляем Flatten
            backbone = nn.Sequential(backbone, nn.Flatten())
            embedding_dim = 2048  # ResNet50 имеет 2048-мерные эмбеддинги
            print("✓ Загружена модель: ResNet50 (torchvision, ImageNet-1K)")
        
        # Заморозка backbone (не обучаем базовую модель)
        for param in backbone.parameters():
            param.requires_grad = False
        
        print(f"✓ Backbone заморожен (не обучается)")
        print(f"✓ Размерность эмбеддингов: {embedding_dim}")
        
        # Создание классификационной головы
        classifier_head = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, n_classes)
        )
        
        # Полная модель
        class ViTWithHead(nn.Module):
            def __init__(self, backbone, head):
                super().__init__()
                self.backbone = backbone
                self.head = head
            
            def forward(self, x):
                # Извлечение признаков (без градиентов для backbone)
                with torch.no_grad():
                    features = self.backbone(x)
                
                # Классификация (с градиентами для головы)
                logits = self.head(features)
                return logits
        
        self.model = ViTWithHead(backbone, classifier_head)
        
        print(f"✓ Классификационная голова создана: {embedding_dim} -> 512 -> 256 -> {n_classes}")
        print(f"✓ Модель готова к обучению\n")
    
    def train(self, train_paths, train_labels):
        """
        Обучение классификатора на GPU
        
        Args:
            train_paths (list): Пути к обучающим изображениям
            train_labels (list): Метки обучающих изображений
        """
        print("\n" + "="*60)
        print("ОБУЧЕНИЕ VISION TRANSFORMER КЛАССИФИКАТОРА")
        print("="*60)
        
        # Установка устройства (GPU для обучения)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\n🖥  Устройство обучения: {self.device}")
        
        if self.device.type == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Память: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("   ⚠ GPU не найден, обучение будет медленным!")
        
        # Подготовка меток
        self.class_names = sorted(list(set(train_labels)))
        encoded_labels = self.label_encoder.fit_transform(train_labels)
        
        print(f"\n📊 Данные:")
        print(f"   Количество изображений: {len(train_paths)}")
        print(f"   Количество классов: {len(self.class_names)}")
        print(f"   Классы: {self.class_names}")
        
        # Создание модели
        if self.model is None:
            self.create_model(len(self.class_names))
        
        self.model = self.model.to(self.device)
        
        # Создание DataLoader
        dataset = ImageDataset(train_paths, encoded_labels, transform=self.train_transform)
        
        # Разделение на train/val
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Обновляем трансформации для валидации (без аугментации)
        val_dataset.dataset.transform = self.test_transform
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0  # Windows часто имеет проблемы с num_workers > 0
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        print(f"\n📦 Батчи:")
        print(f"   Размер батча: {self.batch_size}")
        print(f"   Train батчей: {len(train_loader)}")
        print(f"   Val батчей: {len(val_loader)}")
        
        # Оптимизатор и функция потерь
        # Обучаем только классификационную голову
        optimizer = optim.Adam(
            self.model.head.parameters(),  # Только голова!
            lr=self.learning_rate
        )
        criterion = nn.CrossEntropyLoss()
        
        # Scheduler для уменьшения learning rate
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3
        )
        
        print(f"\n⚙  Параметры обучения:")
        print(f"   Learning rate: {self.learning_rate}")
        print(f"   Epochs: {self.epochs}")
        print(f"   Scheduler: ReduceLROnPlateau (patience=3, factor=0.5)")
        print(f"   Оптимизатор: Adam (только голова)")
        
        # Обучение
        best_val_acc = 0.0
        train_losses = []
        val_accuracies = []
        
        print(f"\n{'='*60}")
        print("НАЧАЛО ОБУЧЕНИЯ")
        print(f"{'='*60}\n")
        
        for epoch in range(self.epochs):
            print(f"Эпоха {epoch+1}/{self.epochs}")
            print("-" * 60)
            
            # === ОБУЧЕНИЕ ===
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            train_bar = tqdm(train_loader, desc="Training", leave=False)
            for images, labels in train_bar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Статистика
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                train_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100 * correct / total:.2f}%'
                })
            
            epoch_loss = running_loss / len(train_loader)
            train_acc = 100 * correct / total
            train_losses.append(epoch_loss)
            
            # === ВАЛИДАЦИЯ ===
            self.model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                val_bar = tqdm(val_loader, desc="Validation", leave=False)
                for images, labels in val_bar:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = self.model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    
                    val_bar.set_postfix({
                        'acc': f'{100 * val_correct / val_total:.2f}%'
                    })
            
            val_acc = 100 * val_correct / val_total
            val_accuracies.append(val_acc)
            
            # Обновление learning rate
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_acc)
            new_lr = optimizer.param_groups[0]['lr']
            
            # Вывод статистики
            print(f"Train Loss: {epoch_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
            
            # Если LR изменился, выводим уведомление
            if new_lr != old_lr:
                print(f"⚠ Learning rate снижен: {old_lr:.6f} → {new_lr:.6f}")
            
            # Сохранение лучшей модели
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model()
                print(f"✓ Модель сохранена (лучшая val acc: {best_val_acc:.2f}%)")
            
            print()
        
        print(f"{'='*60}")
        print("ОБУЧЕНИЕ ЗАВЕРШЕНО")
        print(f"{'='*60}")
        print(f"Лучшая точность на валидации: {best_val_acc:.2f}%")
        print()
    
    def test(self, test_paths, test_labels=None):
        """
        Тестирование классификатора на CPU
        
        Args:
            test_paths (list): Пути к тестовым изображениям
            test_labels (list, optional): Метки тестовых изображений
        
        Returns:
            tuple: Предсказанные метки и точность
        """
        print("\n" + "="*60)
        print("ТЕСТИРОВАНИЕ VISION TRANSFORMER КЛАССИФИКАТОРА")
        print("="*60)
        
        # Установка устройства (CPU для инференса)
        self.device = torch.device('cpu')
        print(f"\n🖥  Устройство инференса: {self.device}")
        
        if self.model is None:
            self.load_model()
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Создание DataLoader
        # Для тестирования используем dummy labels если их нет
        dummy_labels = [0] * len(test_paths) if test_labels is None else self.label_encoder.transform(test_labels)
        
        dataset = ImageDataset(test_paths, dummy_labels, transform=self.test_transform)
        test_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        print(f"\n📊 Данные:")
        print(f"   Количество изображений: {len(test_paths)}")
        print(f"   Количество батчей: {len(test_loader)}")
        
        # Инференс
        all_predictions = []
        
        print(f"\n{'='*60}")
        print("НАЧАЛО ИНФЕРЕНСА")
        print(f"{'='*60}\n")
        
        with torch.no_grad():
            test_bar = tqdm(test_loader, desc="Testing")
            for images, _ in test_bar:
                images = images.to(self.device)
                
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
        
        # Декодирование предсказаний
        predictions = self.label_encoder.inverse_transform(all_predictions)
        
        # Оценка качества
        accuracy = None
        if test_labels is not None:
            accuracy = accuracy_score(test_labels, predictions)
            
            print(f"\n{'='*60}")
            print("РЕЗУЛЬТАТЫ")
            print(f"{'='*60}")
            print(f"\n✓ Точность классификации: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
            
            # Подробный отчет
            report = classification_report(
                test_labels, predictions,
                target_names=self.class_names,
                digits=4
            )
            print("Отчет по классам:")
            print(report)
        
        return predictions, accuracy
    
    def save_model(self):
        """Сохранение модели в файл"""
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        
        # Сохранение весов модели
        model_path = os.path.join(self.model_dir, 'vit_model.pth')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'class_names': self.class_names,
            'label_encoder_classes': self.label_encoder.classes_.tolist(),
            'image_size': self.image_size
        }, model_path)
        
        # Сохранение метаданных в JSON
        metadata = {
            'class_names': self.class_names,
            'image_size': self.image_size,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'model_type': 'DINOv2-ViT-Small'
        }
        
        metadata_path = os.path.join(self.model_dir, 'metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Модель сохранена: {model_path}")
    
    def load_model(self):
        """Загрузка модели из файла"""
        model_path = os.path.join(self.model_dir, 'vit_model.pth')
        
        if not os.path.exists(model_path):
            abs_path = os.path.abspath(model_path)
            raise ValueError(
                f"Модель не найдена по пути: {abs_path}\n"
                f"Проверьте, что файл vit_model.pth существует в директории: "
                f"{os.path.abspath(self.model_dir)}"
            )
        
        print(f"Загрузка модели из {model_path}...")
        
        # Загрузка checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Восстановление метаданных
        self.class_names = checkpoint['class_names']
        self.image_size = checkpoint.get('image_size', self.image_size)
        
        # Восстановление label encoder
        self.label_encoder = LabelEncoder()
        self.label_encoder.classes_ = np.array(checkpoint['label_encoder_classes'])
        
        # Создание модели
        if self.model is None:
            self.create_model(len(self.class_names))
        
        # Загрузка весов
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"✓ Модель загружена успешно")
        print(f"  Классы: {self.class_names}")
