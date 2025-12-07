#!/usr/bin/env python3
"""
Демонстрационный скрипт для показа работы всех фильтров.
"""

import cv2 as cv
import numpy as np
import os
import sys

# Добавляем текущую директорию в путь для импорта
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from script import (
    resize_image, sepia_filter, vignette_filter, pixelate_region,
    add_rectangular_frame, add_decorative_frame, add_lens_flare,
    add_watercolor_texture, interactive_pixelate
)

def show_result(title, image, original=None):
    """Показывает результат фильтра рядом с оригиналом."""
    if original is not None:
        cv.imshow('Исходное', original)
        cv.imshow(title, image)
    else:
        cv.imshow(title, image)
    
    print(f"   >> Показ: {title}. Нажмите любую клавишу для продолжения...")
    cv.waitKey(0)
    cv.destroyAllWindows()

def demo_all_filters():
    """Демонстрирует работу всех фильтров."""
    print("Загрузка тестового изображения из файла...")
    image_path = "images/test_image_1.jpg"
    test_image = cv.imread(image_path)
    if test_image is None:
        print(f"Ошибка: не удалось загрузить изображение {image_path}")
        print("Создание тестового изображения...")
        test_image = create_test_image()
    
    os.makedirs('demo_results', exist_ok=True)
    
    print("Демонстрация фильтров:")
    print("=" * 60)
    
    print("0. Исходное изображение...")
    cv.imwrite('demo_results/0_original.jpg', test_image)
    show_result('0. Исходное изображение', test_image)
    
    print("\n1. Изменение разрешения...")
    resized = resize_image(test_image, 600, 400)
    cv.imwrite('demo_results/1_resize.jpg', resized)
    show_result('1. Изменение разрешения (600x400)', resized, test_image)
    
    print("\n2. Эффект сепии...")
    sepia = sepia_filter(test_image)
    cv.imwrite('demo_results/2_sepia.jpg', sepia)
    show_result('2. Эффект сепии', sepia, test_image)
    
    print("\n3. Эффект виньетки...")
    vignette = vignette_filter(test_image, 0.7)
    cv.imwrite('demo_results/3_vignette.jpg', vignette)
    show_result('3. Эффект виньетки (сила 0.7)', vignette, test_image)
    
    print("\n4. Пикселизация области (интерактивно)...")
    pixelated = interactive_pixelate(test_image)
    cv.imwrite('demo_results/4_pixelate.jpg', pixelated)
    show_result('4. Пикселизация области', pixelated, test_image)
    

    print("\n5. Прямоугольная рамка...")
    framed = add_rectangular_frame(test_image, 20, (0, 255, 0))
    cv.imwrite('demo_results/5_frame.jpg', framed)
    show_result('5. Прямоугольная рамка (зелёная)', framed, test_image)
    
    print("\n6. Декоративная рамка (волнистая)...")
    decorative = add_decorative_frame(test_image, "wavy", 25, (255, 0, 0))
    cv.imwrite('demo_results/6_decorative_wavy.jpg', decorative)
    show_result('6. Декоративная рамка (волнистая)', decorative, test_image)
    
    print("\n7. Декоративная рамка (узорная)...")
    decorative_pattern = add_decorative_frame(test_image, "pattern", 25, (0, 0, 255))
    cv.imwrite('demo_results/7_decorative_pattern.jpg', decorative_pattern)
    show_result('7. Декоративная рамка (узорная)', decorative_pattern, test_image)
    
    print("\n7.1. Декоративная рамка (прямая)...")
    decorative_straight = add_decorative_frame(test_image, "straight", 25, (255, 0, 255))
    cv.imwrite('demo_results/7_1_decorative_straight.jpg', decorative_straight)
    show_result('7.1. Декоративная рамка (прямая)', decorative_straight, test_image)
    
    print("\n7.2. Декоративная рамка (красная)...")
    decorative_red = add_decorative_frame(test_image, "red", 25, (0, 255, 255))
    cv.imwrite('demo_results/7_2_decorative_red.jpg', decorative_red)
    show_result('7.2. Декоративная рамка (красная)', decorative_red, test_image)
    
    print("\n7.3. Декоративная рамка (цветочная)...")
    decorative_floral = add_decorative_frame(test_image, "floral", 25, (255, 255, 0))
    cv.imwrite('demo_results/7_3_decorative_floral.jpg', decorative_floral)
    show_result('7.3. Декоративная рамка (цветочная)', decorative_floral, test_image)
    
    print("\n8. Эффект бликов...")
    flared = add_lens_flare(test_image, 200, 150, 0.8)
    cv.imwrite('demo_results/8_flare.jpg', flared)
    show_result('8. Эффект бликов', flared, test_image)
    
    print("\n9. Текстура акварельной бумаги...")
    watercolor = add_watercolor_texture(test_image, 0.8)
    cv.imwrite('demo_results/9_watercolor.jpg', watercolor)
    show_result('9. Текстура акварельной бумаги', watercolor, test_image)
    
    print("\n" + "=" * 60)
    print("✓ Демонстрация завершена!")
    print("✓ Все результаты сохранены в папке 'demo_results/'")
    print("=" * 60)


def show_help():
    """Показывает справку по использованию."""
    print("""
Демонстрационный скрипт для библиотеки фильтров изображений

Использование:
    python demo.py              - запустить демонстрацию всех фильтров
    python demo.py --help       - показать эту справку

Демонстрация показывает каждый фильтр последовательно (нажмите любую клавишу для продолжения).
Все результаты сохраняются в папку 'demo_results' с примерами работы всех фильтров:
  0. Исходное изображение
  1. Изменение разрешения
  2. Эффект сепии
  3. Эффект виньетки
  4. Пикселизация области
  5. Прямоугольная рамка
  6. Декоративная рамка (волнистая)
  7. Декоративная рамка (узорная)
  7.1. Декоративная рамка (прямая)
  7.2. Декоративная рамка (красная)
  7.3. Декоративная рамка (цветочная)
  8. Эффект бликов
  9. Текстура акварельной бумаги

Для работы с собственными изображениями используйте:
    python script.py -i <путь_к_изображению> -f <тип_фильтра> [параметры]
""")


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h', 'help']:
        show_help()
    else:
        demo_all_filters()
