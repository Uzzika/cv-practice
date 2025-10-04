import cv2
import numpy as np
import math

# ===== ФУНКЦИЯ ИЗМЕНЕНИЯ РАЗРЕШЕНИЯ =====
def change_resolution(image, new_width, new_height):
    """
    Изменяет разрешение изображения методом ближайшего соседа
    
    Параметры:
    image - исходное изображение
    new_width - новая ширина
    new_height - новая высота
    
    Возвращает:
    resized_image - изображение с новым разрешением
    """
    old_height, old_width = image.shape[:2]

    # Вычисляем коэффициенты масштабирования
    scale_x = old_width / new_width
    scale_y = old_height / new_height

    # Создаем пустое изображение нового размера
    if len(image.shape) == 3:
        resized_image = np.zeros((new_height, new_width, image.shape[2]), dtype=image.dtype)
    else:
        resized_image = np.zeros((new_height, new_width), dtype=image.dtype)

    # Заполняем новое изображение пикселями из исходного
    for y in range(new_height):
        for x in range(new_width):
            # Вычисляем координаты в исходном изображении
            old_x = int(x * scale_x)
            old_y = int(y * scale_y)

            # Проверяем границы
            old_x = min(old_x, old_width - 1)
            old_y = min(old_y, old_height - 1)

            # Копируем пиксель
            resized_image[y, x] = image[old_y, old_x]
    
    return resized_image

# ===== ДЕМОНСТРАЦИЯ ФУНКЦИЙ =====
if __name__ == "__main__":
    # Загружаем изображение
    image = cv2.imread('test_image.jpg')

    if image is not None:
        print("Изображение успешно загружено!")
        print(f"Размер исходного изображения: {image.shape}")
        
        # Конвертируем в RGB для корректной работы с цветами
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Демонстрация изменения разрешения
        print("\n=== Изменение разрешения ===")
        resized = change_resolution(image_rgb, 300, 200)
        print(f"Размер после изменения: {resized.shape}")
        
        # Демонстрация эффекта сепии
        print("\n=== Эффект сепии ===")
        sepia = apply_sepia(image_rgb)
        print("Эффект сепии применен!")
        
        # Показываем результаты
        cv2.imshow('Original', image)  # Исходное (BGR)
        cv2.imshow('Resized', cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))
        cv2.imshow('Sepia', cv2.cvtColor(sepia, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Ошибка: не удалось загрузить изображение!")