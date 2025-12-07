"""
Скрипт для демонстрации всех фильтров
Автоматически применяет все фильтры и сохраняет результаты
"""
import cv2
import sys
import os

# Добавляем текущую директорию в путь
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Импортируем функции из main.py
from main import (
    resize_image, apply_sepia, apply_vignette, pixelate_region,
    add_simple_frame, add_decorative_frame, add_lens_flare, add_watercolor_texture
)


def demo_all_filters(image_path):
    """Демонстрация всех фильтров"""
    
    # Загружаем изображение
    image = cv2.imread(image_path)
    if image is None:
        print(f"Ошибка: не удалось загрузить {image_path}")
        return
    
    print(f"Загружено изображение: {image.shape}")
    
    # Создаем папку для результатов
    output_dir = "filter_results"
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    # 1. Изменение разрешения
    print("1. Применяем изменение разрешения...")
    results['resize'] = resize_image(image, scale_factor=0.7)
    cv2.imwrite(f"{output_dir}/01_resize.jpg", results['resize'])
    
    # 2. Сепия
    print("2. Применяем эффект сепии...")
    results['sepia'] = apply_sepia(image, intensity=0.9)
    cv2.imwrite(f"{output_dir}/02_sepia.jpg", results['sepia'])
    
    # 3. Виньетка
    print("3. Применяем эффект виньетки...")
    results['vignette'] = apply_vignette(image, intensity=0.6)
    cv2.imwrite(f"{output_dir}/03_vignette.jpg", results['vignette'])
    
    # 4. Пикселизация (центральная область)
    print("4. Применяем пикселизацию...")
    h, w = image.shape[:2]
    results['pixelate'] = pixelate_region(
        image, 
        w//4, h//4,  # Левый верхний угол
        3*w//4, 3*h//4,  # Правый нижний угол
        pixel_size=15
    )
    cv2.imwrite(f"{output_dir}/04_pixelate.jpg", results['pixelate'])
    
    # 5. Простая рамка
    print("5. Применяем простую рамку...")
    results['simple_frame'] = add_simple_frame(image, frame_width=25, color=(0, 215, 255))
    cv2.imwrite(f"{output_dir}/05_simple_frame.jpg", results['simple_frame'])
    
    # 6. Фигурная рамка (rounded)
    print("6. Применяем фигурную рамку (скругленные углы)...")
    results['decorative_rounded'] = add_decorative_frame(
        image, frame_width=30, color=(128, 64, 0), frame_type='rounded'
    )
    cv2.imwrite(f"{output_dir}/06_decorative_rounded.jpg", results['decorative_rounded'])
    
    # 7. Фигурная рамка (wave)
    print("7. Применяем фигурную рамку (волны)...")
    results['decorative_wave'] = add_decorative_frame(
        image, frame_width=25, color=(255, 0, 0), frame_type='wave'
    )
    cv2.imwrite(f"{output_dir}/07_decorative_wave.jpg", results['decorative_wave'])
    
    # 8. Фигурная рамка (zigzag)
    print("8. Применяем фигурную рамку (зигзаг)...")
    results['decorative_zigzag'] = add_decorative_frame(
        image, frame_width=25, color=(0, 0, 255), frame_type='zigzag'
    )
    cv2.imwrite(f"{output_dir}/08_decorative_zigzag.jpg", results['decorative_zigzag'])
    
    # 9. Блики объектива
    print("9. Применяем блики объектива...")
    results['lens_flare'] = add_lens_flare(image, intensity=0.8)
    cv2.imwrite(f"{output_dir}/09_lens_flare.jpg", results['lens_flare'])
    
    # 10. Текстура акварельной бумаги
    print("10. Применяем текстуру акварельной бумаги...")
    results['watercolor'] = add_watercolor_texture(image, intensity=0.4)
    cv2.imwrite(f"{output_dir}/10_watercolor.jpg", results['watercolor'])
    
    # Комбинированные эффекты
    print("\n11. Создаем комбинированные эффекты...")
    
    # Винтажный эффект (сепия + виньетка + рамка)
    vintage = apply_sepia(image, intensity=0.8)
    vintage = apply_vignette(vintage, intensity=0.5)
    vintage = add_simple_frame(vintage, frame_width=20, color=(50, 30, 10))
    cv2.imwrite(f"{output_dir}/11_vintage_combo.jpg", vintage)
    
    # Художественный эффект (акварель + рамка)
    artistic = add_watercolor_texture(image, intensity=0.35)
    artistic = add_decorative_frame(artistic, frame_width=30, color=(100, 80, 60), frame_type='rounded')
    cv2.imwrite(f"{output_dir}/12_artistic_combo.jpg", artistic)
    
    print(f"\n✓ Все фильтры применены успешно!")
    print(f"✓ Результаты сохранены в папку '{output_dir}/'")
    
    # Показываем сравнение оригинала и нескольких фильтров
    print("\nОтображаем результаты (нажмите любую клавишу для продолжения)...\n")
    
    # Создаем сетку изображений для сравнения
    scale = 0.3  # Масштаб для отображения
    target_h = int(image.shape[0] * scale)
    target_w = int(image.shape[1] * scale)
    
    def resize_for_display(img):
        return cv2.resize(img, (target_w, target_h))
    
    # Первая строка: оригинал, resize, sepia, vignette
    row1 = cv2.hconcat([
        resize_for_display(image),
        resize_for_display(results['resize']),
        resize_for_display(results['sepia']),
        resize_for_display(results['vignette'])
    ])
    
    # Вторая строка: pixelate, simple_frame, decorative, lens_flare
    row2 = cv2.hconcat([
        resize_for_display(results['pixelate']),
        resize_for_display(results['simple_frame']),
        resize_for_display(results['decorative_rounded']),
        resize_for_display(results['lens_flare'])
    ])
    
    # Третья строка: watercolor, vintage, artistic, decorative_wave
    row3 = cv2.hconcat([
        resize_for_display(results['watercolor']),
        resize_for_display(vintage),
        resize_for_display(artistic),
        resize_for_display(results['decorative_wave'])
    ])
    
    # Объединяем все строки
    grid = cv2.vconcat([row1, row2, row3])
    
    # Добавляем подписи
    font = cv2.FONT_HERSHEY_SIMPLEX
    labels = [
        ['Original', 'Resize', 'Sepia', 'Vignette'],
        ['Pixelate', 'Simple Frame', 'Decorative', 'Lens Flare'],
        ['Watercolor', 'Vintage', 'Artistic', 'Wave Frame']
    ]
    
    grid_with_labels = grid.copy()
    for row_idx, row_labels in enumerate(labels):
        for col_idx, label in enumerate(row_labels):
            x = int(col_idx * image.shape[1] * scale + 10)
            y = int(row_idx * image.shape[0] * scale + 30)
            cv2.putText(grid_with_labels, label, (x, y), font, 0.5, (255, 255, 255), 2)
            cv2.putText(grid_with_labels, label, (x, y), font, 0.5, (0, 0, 0), 1)
    
    cv2.imshow('All Filters Demo', grid_with_labels)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("\nГотово! Проверьте папку filter_results/ для просмотра всех результатов.")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Использование: python demo_filters.py <путь_к_изображению>")
        print("Пример: python demo_filters.py test_image.jpg")
        sys.exit(1)
    
    demo_all_filters(sys.argv[1])
