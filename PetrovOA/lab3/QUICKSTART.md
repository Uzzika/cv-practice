# Быстрый старт - Vision Transformer классификатор

## Шаг 1: Установка зависимостей

### Базовые зависимости
```bash
pip install opencv-python numpy scikit-learn joblib
```

### PyTorch (для GPU - CUDA 11.8)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### PyTorch (для CPU - если нет GPU)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Дополнительные зависимости
```bash
pip install -r requirements_pytorch.txt
```

## Шаг 2: Подготовка данных

Создайте train/test split (80/20):

```bash
python create_split.py
```

Результат:
- `data/train_nnsu.txt` - обучающая выборка (~130 изображений)
- `data/test_nnsu.txt` - тестовая выборка (~34 изображения)

## Шаг 3: Обучение модели

```bash
python main.py --data_dir data --train_file data/train_nnsu.txt --test_file data/test_nnsu.txt --mode train --algorithm vit --epochs 20
```

**Параметры:**
- `--epochs 20` - количество эпох (можно увеличить до 30-50 для лучшего качества)
- `--batch_size 16` - размер батча (уменьшите до 8, если не хватает GPU памяти)
- `--learning_rate 0.001` - скорость обучения

**Ожидаемое время:** 5-7 минут на GPU, ~20-30 минут на CPU

## Шаг 4: Тестирование модели

```bash
python main.py --data_dir data --train_file data/train_nnsu.txt --test_file data/test_nnsu.txt --mode test --algorithm vit
```

**Ожидаемая точность:** 95-97%

## Шаг 5: Обучение + Тестирование (в одной команде)

```bash
python main.py --data_dir data --train_file data/train_nnsu.txt --test_file data/test_nnsu.txt --mode both --algorithm vit --epochs 20
```

---

## Сравнение с BoW

### Обучение BoW (быстро, для сравнения)

```bash
python main.py --data_dir data --train_file data/train.txt --test_file data/test.txt --mode both --algorithm bow
```

### Сравнение обоих методов

```bash
python main.py --data_dir data --train_file data/train_nnsu.txt --test_file data/test_nnsu.txt --mode both --algorithm both --epochs 20
```

---

## Решение проблем

### Проблема: "Количество изображений: 0"

**Причина:** Неправильный путь к данным

**Решение:** 
- Используйте `--data_dir data` (не `data/NNSUDataset`)
- Убедитесь, что пути в train/test файлах начинаются с `NNSUDataset/...`

### Проблема: "CUDA out of memory"

**Причина:** Не хватает GPU памяти

**Решение:**
```bash
python main.py --data_dir data --train_file data/train_nnsu.txt --test_file data/test_nnsu.txt --mode train --algorithm vit --batch_size 8 --epochs 20
```

### Проблема: Загрузка модели зависает

**Причина:** Загрузка весов из интернета может занять время

**Решение:** 
- Подождите - первая загрузка ViT-B/16 занимает 10-60 секунд
- Модель загружается из официального источника PyTorch (надежно)
- При повторных запусках модель берется из кэша (мгновенно)

**Проверка загрузки:**
```python
import torch
from torchvision.models import vit_b_16, ViT_B_16_Weights
# Это загрузит модель в кэш (~330 MB)
model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
print("✓ Модель загружена успешно!")
```

### Проблема: Медленное обучение на CPU

**Причина:** Трансформеры требовательны к вычислениям

**Решение:**
- Используйте GPU для обучения
- Уменьшите batch_size до 4-8
- Уменьшите количество эпох до 10

---

## Структура сохраненной модели

После обучения в директории `vit_model/` будут созданы:

```
vit_model/
├── vit_model.pth      # Веса модели (PyTorch checkpoint)
└── metadata.json      # Метаданные (классы, параметры)
```

Модель можно использовать повторно без переобучения:

```bash
python main.py --data_dir data --train_file data/train_nnsu.txt --test_file data/test_nnsu.txt --mode test --algorithm vit
```

---

## Визуализация SIFT (для BoW)

```bash
python main.py --visualize "data\NNSUDataset\01_NizhnyNovgorodKremlin\1e1f8c18-6a6b-41e5-bd48-abb3be5e388a.jpg"
```

---

## Проверка наличия GPU

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

---

## Дополнительная информация

- Полная документация: `README.md`
- Подробное руководство по ViT: `VIT_GUIDE.md`
