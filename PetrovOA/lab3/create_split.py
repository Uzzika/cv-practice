"""
Скрипт для создания train/test split из NNSUDataset
Разделение: 80% train, 20% test
"""
import os
import random
from pathlib import Path

# Установка seed для воспроизводимости
random.seed(42)

# Путь к данным
data_dir = Path("data/NNSUDataset")
output_dir = Path("data")

# Получаем все изображения
image_extensions = {'.jpg', '.jpeg', '.png'}
all_images = []

for class_dir in sorted(data_dir.iterdir()):
    if class_dir.is_dir():
        class_images = []
        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() in image_extensions:
                # Относительный путь от data/
                rel_path = img_path.relative_to(Path("data"))
                class_images.append(str(rel_path).replace('\\', '/'))
        
        # Перемешиваем изображения класса
        random.shuffle(class_images)
        
        # Разделяем 80/20
        split_idx = int(len(class_images) * 0.8)
        train_images = class_images[:split_idx]
        test_images = class_images[split_idx:]
        
        print(f"Класс {class_dir.name}:")
        print(f"  Train: {len(train_images)} изображений")
        print(f"  Test: {len(test_images)} изображений")
        
        all_images.append({
            'class': class_dir.name,
            'train': train_images,
            'test': test_images
        })

# Собираем все train и test
all_train = []
all_test = []

for class_data in all_images:
    all_train.extend(class_data['train'])
    all_test.extend(class_data['test'])

# Перемешиваем
random.shuffle(all_train)
random.shuffle(all_test)

# Записываем в файлы
train_file = output_dir / "train_nnsu.txt"
test_file = output_dir / "test_nnsu.txt"

with open(train_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(all_train))

with open(test_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(all_test))

print(f"\n✓ Создано:")
print(f"  {train_file}: {len(all_train)} изображений")
print(f"  {test_file}: {len(all_test)} изображений")
print(f"\nИспользуйте команду:")
print(f"python main.py --data_dir data --train_file data/train_nnsu.txt --test_file data/test_nnsu.txt --algorithm vit")
