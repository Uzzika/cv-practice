# Лабораторная работа №3. Классификация изображений с использованием библиотеки OpenCV

## Как устроено решение

### 1) BoW (мешок слов)
1. **SIFT-дескрипторы**: для каждого изображения извлекаются до `max_kp` локальных SIFT
2. **Словарь (KMeans, k центров)**: все train-дескрипторы объединяются и кластеризуются
3. **Гистограммы**: для изображения считаем частоты попаданий дескрипторов в k-кластеров, L2-нормируем
4. **Классификация**: LinearSVC по гистограммам (признаковое пространство размера k)


### 2) CNN (ResNet50, перенос обучения)
1. Загружается предобученный **ResNet50** (`IMAGENET1K_V2`), **заменяется последний слой** `model.fc` на `Linear(in_features, num_classes)`
2. Препроцессинг — **официальные** `weights.transforms()` (Resize → CenterCrop → ToTensor → Normalize)
3. Обучение: Adam + CrossEntropyLoss
4. Сохраняется **лучший чекпойнт** по `val_acc`; для теста веса загружаются и считаются метрики


## Параметры CLI (main.py)

**Общее**
- `--data_dir` — корневая папка с `ExtDataset/` и `NNSUDataset/` (**обязательно**)
- `--train_list` — путь к `train.txt` (**обязательно**)
- `--mode {train|test|both}` — режим
- `--algo {bow|cnn}` — выбор алгоритма

**BoW**
- `--k` — размер словаря (KMeans), по умолчанию `300`
- `--C` — параметр регуляризации `LinearSVC`, по умолчанию `1.0`
- `--max_kp` — максимум SIFT-точек на изображение (по умолчанию `1000`)
- `--vizualize_dir` — каталог для сохранения картинок с нарисованными SIFT-точками
- `--viz_per_class` — сколько картинок на класс визуализировать
- `--model_out` / `--model_in` — файл для сохранения/загрузки модели (pickle)

**CNN**
- `--val_size` — доля валидации из train, по умолчанию `0.2`
- `--epochs` — число эпох (по умолчанию `8`)
- `--batch_size` — размер батча (по умолчанию `16`)
- `--lr` — learning rate Adam (по умолчанию `1e-3`)
- `--device {cuda|cpu}` — устройство
- `--model_out` / `--model_in` — файл весов `.pth` для сохранения/загрузки

---

## Запуск

### BoW: обучение + тест
```bash
python main.py --algo bow --mode both --data_dir ./data --train_list ./data/train.txt --k 300 --C 1.0 --max_kp 1000 --model_out artifacts_bow.pkl
```

Только обучение:
```bash
python main.py --algo bow --mode train --data_dir ./data --train_list ./data/train.txt --k 300 --C 1.0 --max_kp 1000 --model_out artifacts_bow.pkl
```

Только тест (по сохранённой модели):
```bash
python main.py --algo bow --mode test --data_dir ./data --train_list ./data/train.txt --model_in artifacts_bow.pkl --max_kp 1000
```

### CNN (ResNet50): обучение + финальный тест
```bash
python main.py --algo cnn --mode both --data_dir ./data --train_list ./data/train.txt --val_size 0.2 --epochs 12 --batch_size 16 --lr 1e-3 --device cuda --weights_out resnet50.pth
```

Только обучение:
```bash
python main.py --algo cnn --mode train --data_dir ./data --train_list ./data/train.txt --val_size 0.2 --epochs 12 --batch_size 16 --lr 1e-3 --device cuda --weights_out resnet50.pth
```

Только тест:
```bash
python main.py --algo cnn --mode test --data_dir ./data --train_list ./data/train.txt --device cuda --weights_in resnet50.pth
```

## Результаты

### BoW
- Параметры: `k=400`, `C=1.0`, `max_kp=1000`
- **Accuracy:** `0.9518`

### CNN (ResNet50)
- Параметры: `epochs=8`, `batch_size=16`, `lr=1e-3`, `val_size=0.2`, `device=cuda`
- **Accuracy:** `0.9880`
