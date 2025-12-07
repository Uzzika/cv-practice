import os

import cv2
import numpy as np


def sepia_filter(image, intensity=1.0):
    """
    Функция применения фотоэффекта сепии к изображению.

    Args:
        image: исходное изображение
        intensity: интенсивность сепии (0.0 - 1.0)
    """
    sepia_matrix = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ])

    sepia_matrix = sepia_matrix * intensity + np.eye(3) * (1 - intensity)
    sepia_image = cv2.transform(image, sepia_matrix)

    sepia_image = np.clip(sepia_image, 0, 255).astype(np.uint8)

    return sepia_image


def apply_vignette_elliptical(image, scale=1.5, intensity=0.7):
    """
    Функция применения фотоэффекта виньетки к изображению.

    Args:
        image: исходное изображение
        scale: масштаб виньетки
        intensity: интенсивность затемнения краев (0.0 - 1.0)
    """
    height, width = image.shape[:2]

    kernel_x = cv2.getGaussianKernel(width, width / scale)
    kernel_y = cv2.getGaussianKernel(height, height / scale)
    kernel = kernel_y * kernel_x.T

    mask = kernel / kernel.max()
    mask = mask * intensity

    mask = mask[:, :, np.newaxis]

    result = image.astype(np.float32) * mask

    return np.clip(result, 0, 255).astype(np.uint8)


def add_rectangular_border(image, border_width=10, color=(0, 0, 0)):
    """
    Накладывает прямоугольную одноцветную рамку по краям изображения

    Args:
        image: исходное изображение
        border_width: ширина рамки в пикселях
        color: цвет рамки (B, G, R)
    """
    h, w = image.shape[:2]

    if border_width is None:
        border_width = int(min(h, w) / 10)

    image[:border_width] = color
    image[-border_width:] = color
    image[:, 0:border_width] = color
    image[:, -border_width:] = color

    return image


def add_lens_flare(image, flare_position=(0.5, 0.5), intensity=1.0, flare_size=0.3):
    """
    Накладывает текстуру блика на изображение.

    Args:
        image: исходное изображение
        flare_position: позиция блика в нормализованных координатах (0.0-1.0)
        intensity: интенсивность эффекта (0.0 - 1.0)
        flare_size: размер блика (0.1 - 1.0)
    """
    glared_image = image.astype(np.float32)
    h_img, w_img = image.shape[:2]
    
    center_x = int(flare_position[0] * w_img)
    center_y = int(flare_position[1] * h_img)
    
    texture_path = os.path.join('texture', 'flare_texture.jpg')
    
    flare_texture = cv2.imread(texture_path)

    h_glare, w_glare = flare_texture.shape[:2]

    scale_factor = flare_size
    new_w = int(w_glare * scale_factor)
    new_h = int(h_glare * scale_factor)
    flare = cv2.resize(flare_texture, (new_w, new_h))
    h_glare, w_glare = flare.shape[:2]
    
    y_start = center_y - h_glare // 2
    y_end = y_start + h_glare
    x_start = center_x - w_glare // 2
    x_end = x_start + w_glare
    
    img_y_start = max(0, y_start)
    img_y_end = min(h_img, y_end)
    img_x_start = max(0, x_start)
    img_x_end = min(w_img, x_end)
    
    glare_y_start = max(0, -y_start)
    glare_y_end = h_glare - max(0, y_end - h_img)
    glare_x_start = max(0, -x_start)
    glare_x_end = w_glare - max(0, x_end - w_img)
    
    glared_image[img_y_start:img_y_end, img_x_start:img_x_end] += flare[glare_y_start:glare_y_end, glare_x_start:glare_x_end] * intensity

    return np.clip(glared_image, 0, 255).astype(np.uint8)


def add_watercolor_texture(image, texture_intensity=0.3, strength=0.9):
    """
    Накладывает текстуру акварельной бумаги из файла

    Args:
        image: исходное изображение
        texture_intensity: интенсивность текстуры (0.0 - 1.0)
        strength: сила применения текстуры (0.0 - 1.0)
    """
    texture_path = os.path.join('texture', 'water_paper.jpg')
    
    texture = cv2.imread(texture_path)
    
    if texture.shape[:2] != image.shape[:2]:
        texture = cv2.resize(texture, (image.shape[1], image.shape[0]))
    
    texture_gray = np.mean(texture, axis=2)
    texture_mask = 1 - (texture_gray / 255.0)
    texture_mask = texture_mask ** (1 / strength)
    texture_mask = texture_mask[:, :, np.newaxis]
    
    blended = image.astype(np.float32) * (1 - texture_mask * texture_intensity) + texture.astype(np.float32) * (texture_mask * texture_intensity)
    
    return np.clip(blended, 0, 255).astype(np.uint8)


def apply_figure_frame(image, frame_path):
    """
    Наложение фигурной рамки по краям изображения

    Args:
        image: исходное изображение
        frame_path: путь до рамки
    """
    result = image.copy()
    h, w = result.shape[:2]

    frame_path = os.path.join('frames', frame_path)

    if os.path.exists(frame_path):
        frame_img = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)

        if frame_img is not None:
            frame_resized = cv2.resize(frame_img, (w, h))

            frame_bgr = frame_resized[:, :, :3]
            alpha = frame_resized[:, :, 3] / 255.0

            for c in range(3):
                result[:, :, c] = result[:, :, c] * (1 - alpha) + frame_bgr[:, :, c] * alpha

    return result



def draw_green_frame(image, point_1, point_2):
    """
    Рисует зеленую рамку толщиной 1 пиксель по заданным координатам

    Args:
        image: исходное изображение
        point_1: координаты левого верхнего угла области в нормализованных координатах (0.0-1.0)
        point_2: координаты правого нижнего угла области в нормализованных координатах (0.0-1.0)
    """
    h, w = image.shape[:2]

    x1 = int(point_1[0] * w)
    y1 = int(point_1[1] * h)
    x2 = int(point_2[0] * w)
    y2 = int(point_2[1] * h)

    if x1 >= 0 and y1 >= 0 and x2 >= 0 and y2 >= 0 and x2 > x1 and y2 > y1:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)

    return image


def pixelate_region(image, point_1, point_2, pixel_size=10):
    """
    Пикселизация заданной прямоугольной области изображения

    Args:
        image: исходное изображение
        point_1: координаты левого верхнего угла области в нормализованных координатах (0.0-1.0)
        point_2: координаты правого нижнего угла области в нормализованных координатах (0.0-1.0)
        pixel_size: размер пикселя для пикселизации
    """

    h, w = image.shape[:2]

    x1 = int(point_1[0] * w)
    y1 = int(point_1[1] * h)
    x2 = int(point_2[0] * w)
    y2 = int(point_2[1] * h)

    if (x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1 or
            x2 > image.shape[1] or y2 > image.shape[0]):
        return image

    region = image[y1:y2, x1:x2]

    h, w = region.shape[:2]

    small_w = max(1, w // pixel_size)
    small_h = max(1, h // pixel_size)
    small = cv2.resize(region, (small_w, small_h), interpolation=cv2.INTER_LINEAR)

    pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    image[y1:y2, x1:x2] = pixelated

    return image