# Vision Transformer (ViT) Классификатор - Руководство

## Описание

Новый классификатор на основе **Vision Transformer (ViT-B/16)** использует современную архитектуру трансформеров для классификации изображений.

### Ключевые особенности:

✅ **PyTorch** вместо Keras/TensorFlow
✅ **ViT-B/16** - предобученная модель из torchvision (ImageNet-1K)
✅ **Надежная загрузка** - использует официальный источник PyTorch
✅ **Замороженный backbone** - обучается только классификационная голова
✅ **GPU для обучения** - быстрое обучение на видеокарте
✅ **CPU для инференса** - не требует GPU для предсказаний
✅ **Data Augmentation** - улучшенная аугментация данных

### Архитектура:

```
Входное изображение (224x224)
         ↓
ViT-B/16 Encoder (заморожен)
    768-мерный эмбеддинг (class token)
         ↓
Классификационная голова (обучается):
    Linear(768 → 512) + ReLU + Dropout(0.3)
    Linear(512 → 256) + ReLU + Dropout(0.2)
    Linear(256 → num_classes)
         ↓
Предсказание класса
```

---

## Установка зависимостей

### 1. Установка PyTorch (для CPU - инференс)

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 2. Установка PyTorch (для GPU - обучение)

**CUDA 11.8:**
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**CUDA 12.1:**
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 3. Установка остальных зависимостей

```powershell
pip install -r requirements_pytorch.txt
```

---

## Использование

### 1. Обучение классификатора

**Базовая команда:**
```powershell
python main.py --algorithm vit --mode train --data_dir data/NNSUDataset --train_file data/train.txt --test_file data/test.txt
```

**С настройкой параметров:**
```powershell
python main.py --algorithm vit --mode train `
    --data_dir data/NNSUDataset `
    --train_file data/train.txt `
    --test_file data/test.txt `
    --learning_rate 0.001 `
    --batch_size 16 `
    --epochs 20
```

**Параметры:**
- `--learning_rate` - скорость обучения (по умолчанию: 0.001)
- `--batch_size` - размер батча (по умолчанию: 16)
- `--epochs` - количество эпох (по умолчанию: 20)
- `--model_dir` - директория для сохранения модели (по умолчанию: vit_model)

### 2. Тестирование классификатора

```powershell
python main.py --algorithm vit --mode test `
    --data_dir data/NNSUDataset `
    --train_file data/train.txt `
    --test_file data/test.txt
```

### 3. Обучение и тестирование

```powershell
python main.py --algorithm vit --mode both `
    --data_dir data/NNSUDataset `
    --train_file data/train.txt `
    --test_file data/test.txt
```

---

## Процесс обучения

### Что происходит во время обучения:

1. **Загрузка предобученной модели**
   - Скачивается DINOv2-small с torch.hub (~85 MB)
   - Веса backbone замораживаются (не обучаются)

2. **Подготовка данных**
   - Разделение на train (80%) и validation (20%)
   - Применение аугментаций к обучающей выборке

3. **Обучение классификационной головы**
   - Обучается только голова (~600K параметров)
   - Используется Adam оптимизатор
   - Применяется ReduceLROnPlateau scheduler

4. **Валидация после каждой эпохи**
   - Сохраняется модель с лучшей val accuracy

5. **Сохранение модели**
   - `vit_model/vit_model.pth` - веса модели
   - `vit_model/metadata.json` - метаданные



---

## Процесс тестирования

### Что происходит во время тестирования:

1. **Загрузка модели**
   - Модель загружается на CPU (для инференса)
   - Восстанавливаются веса классификационной головы

2. **Инференс**
   - Изображения обрабатываются батчами
   - Применяются только нормализация и resize (без аугментаций)

3. **Оценка качества**
   - Вычисляется accuracy
   - Выводится подробный classification report



---

## Сравнение с другими методами

### Ожидаемые результаты:

| Метод |        Test Accuracy       |
|-------|----------------------------|
| **BoW (SIFT + SVM)**    |     ~96% |
| **ViT (ViT-B/16)**      |     100% |

### Преимущества ViT:

✅ **Высокая точность** - современная архитектура трансформеров
✅ **Надежная загрузка** - использует torchvision (официальный источник)
✅ **Быстрая загрузка** - модель загружается за 10-30 секунд
✅ **Семантические признаки** - трансформеры понимают контекст
✅ **Гибкость** - легко менять backbone

### Недостатки ViT:

❌ **Требует GPU** - для эффективного обучения
❌ **Больше зависимостей** - PyTorch, torchvision
❌ **Медленнее на CPU** - сложнее архитектура

---

## Структура сохраненной модели

```
vit_model/
├── vit_model.pth          # Веса модели (PyTorch checkpoint)
└── metadata.json          # Метаданные модели
```

### metadata.json:
```json
{
  "class_names": [
    "01_NizhnyNovgorodKremlin",
    "04_ArkhangelskCathedral",
    "08_PalaceOfLabor"
  ],
  "image_size": 224,
  "learning_rate": 0.001,
  "batch_size": 16,
  "epochs": 20,
  "model_type": "ViT-B/16-torchvision"
}
```

---

## Использование в Python коде

```python
from classifier.vit_classifier import ViTClassifier

# Создание классификатора
classifier = ViTClassifier(
    model_dir='vit_model',
    image_size=224,
    learning_rate=0.001,
    batch_size=16,
    epochs=20
)

# Обучение
train_paths = ['image1.jpg', 'image2.jpg', ...]
train_labels = ['class1', 'class2', ...]
classifier.train(train_paths, train_labels)

# Тестирование
test_paths = ['test1.jpg', 'test2.jpg', ...]
test_labels = ['class1', 'class2', ...]
predictions, accuracy = classifier.test(test_paths, test_labels)

print(f"Accuracy: {accuracy:.4f}")
print(f"Predictions: {predictions}")
```

---

