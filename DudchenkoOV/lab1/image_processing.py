import cv2
import numpy as np
import math
import os

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

    scale_x = old_width / new_width
    scale_y = old_height / new_height

    if len(image.shape) == 3:
        resized_image = np.zeros((new_height, new_width, image.shape[2]), dtype=image.dtype)
    else:
        resized_image = np.zeros((new_height, new_width), dtype=image.dtype)

    for y in range(new_height):
        for x in range(new_width):
            old_x = int(x * scale_x)
            old_y = int(y * scale_y)

            old_x = min(old_x, old_width - 1)
            old_y = min(old_y, old_height - 1)

            resized_image[y, x] = image[old_y, old_x]
    
    return resized_image

def apply_sepia(image_rgb):
    """
    Применяет фотоэффект сепии к изображению
    
    Параметры:
    image - исходное изображение в формате RGB
    
    Возвращает:
    sepia_image - изображение с эффектом сепии
    """
    sepia_image = image_rgb.copy().astype(np.float32)
    
    red = sepia_image[:, :, 0]
    green = sepia_image[:, :, 1]
    blue = sepia_image[:, :, 2]
    
    new_red = red * 0.393 + green * 0.769 + blue * 0.189
    new_green = red * 0.349 + green * 0.686 + blue * 0.168
    new_blue = red * 0.272 + green * 0.534 + blue * 0.131
    
    new_red = np.clip(new_red, 0, 255)
    new_green = np.clip(new_green, 0, 255)
    new_blue = np.clip(new_blue, 0, 255)
    
    sepia_image[:, :, 0] = new_red
    sepia_image[:, :, 1] = new_green
    sepia_image[:, :, 2] = new_blue

    return sepia_image.astype(np.uint8)

# ===== ФУНКЦИЯ ЭФФЕКТА ВИНЬЕТКИ =====
def apply_vignette(image, strength=0.8):
    """
    Применяет эффект виньетки (затемнение по краям изображения)
    
    Параметры:
    image - исходное изображение
    strength - сила эффекта (0-1), где 1 - максимальное затемнение
    
    Возвращает:
    vignette_image - изображение с эффектом виньетки
    """

    vignette_image = image.copy().astype(np.float32)

    height, width = image.shape[:2]

    x = np.arange(width)
    y = np.arange(height)

    # np.meshgrid создает две матрицы координат:
    # x_grid - каждая строка содержит x-координаты
    # y_grid - каждый столбец содержит y-координаты
    x_grid, y_grid = np.meshgrid(x, y)

    x_normalized = (x_grid - width / 2) / (width / 2)
    y_normalized = (y_grid - height / 2) / (height / 2)

    distance = np.sqrt(x_normalized ** 2 + y_normalized ** 2)
    distance = np.clip(distance, 0, 1)

    vignette_mask = 1 - distance * strength

    if len(image.shape) == 3:
        for channel in range(3):
            vignette_image[:, :, channel] *= vignette_mask
    else:
        vignette_mask *= vignette_mask

    vignette_image = np.clip(vignette_image, 0, 255)

    return vignette_image.astype(np.uint8)

def apply_pixelation(image, x, y, width, height, pixel_size = 10):
    """
    Применяет пикселизацию к заданной прямоугольной области изображения
    
    Параметры:
    image - исходное изображение
    x, y - координаты верхнего левого угла области
    width, height - ширина и высота области
    pixel_size - размер пикселя (чем больше, тем сильнее пикселизация)
    
    Возвращает:
    pixelated_image - изображение с пикселизированной областью
    """
    pixelated_image = image.copy()

    x = max(0, min(x, image.shape[1] - 1))
    y = max(0, min(y, image.shape[0] - 1))
    width = min(width, image.shape[1] - x)
    height = min(height, image.shape[0] - y)

    region = pixelated_image[y:y + height, x:x + width]

    for i in range(0, height, pixel_size):
        for j in range(0, width, pixel_size):
            block_height = min(pixel_size, height - i)
            block_width = min(pixel_size, width - j)

            block = region[i:i + block_height, j:j + block_width]

            if len(image.shape) == 3:
                avg_color = np.mean(block, axis =(0, 1))
            else:
                avg_color = np.mean(block)

            region[i:i + block_height, j:j + block_width] = avg_color
    
    return pixelated_image

# ===== ДЕМОНСТРАЦИЯ ФУНКЦИЙ =====
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, 'test_image.jpg')

    image = cv2.imread(image_path)

    if image is not None:
        print("Изображение успешно загружено!")
        print(f"Размер исходного изображения: {image.shape}")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        print("\n=== Изменение разрешения ===")
        resized = change_resolution(image_rgb, 300, 200)
        print(f"Размер после изменения: {resized.shape}")
        
        print("\n=== Эффект сепии ===")
        sepia = apply_sepia(image_rgb)
        print("Эффект сепии применен!")
        
        print("\n=== Эффект виньетки ===")
        vignette = apply_vignette(image_rgb)
        print("Эффект виньетки применен!")

        print("\n=== Пикселизация области ===")
        height, width = image_rgb.shape[:2]
        center_x, center_y = width // 2, height // 2
        pixelated = apply_pixelation(image_rgb, 
                                   center_x - 50, center_y - 50, 
                                   200, 200, pixel_size=15)
        print("Пикселизация применена!")
        
        cv2.imshow('Original', image)
        cv2.imshow('Resized', cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))
        cv2.imshow('Sepia', cv2.cvtColor(sepia, cv2.COLOR_RGB2BGR))
        cv2.imshow('Vignette', cv2.cvtColor(vignette, cv2.COLOR_RGB2BGR))
        cv2.imshow('Pixelation', cv2.cvtColor(pixelated, cv2.COLOR_RGB2BGR))

        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Ошибка: не удалось загрузить изображение!")