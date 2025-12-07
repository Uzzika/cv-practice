import cv2
import numpy as np
import argparse
import sys

# Глобальные переменные для обработки выбора области
selection_start = None
selection_end = None
selecting = False
current_image = None


def mouse_callback(event, x, y, flags, param):
    global selection_start, selection_end, selecting, current_image

    if event == cv2.EVENT_LBUTTONDOWN:
        selection_start = (x, y)
        selection_end = (x, y)
        selecting = True

    elif event == cv2.EVENT_MOUSEMOVE and selecting:
        selection_end = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        selection_end = (x, y)
        selecting = False


def change_resolution(image, new_width=None, new_height=None):
    height, width = image.shape[:2]

    if new_width is None and new_height is None:
        return image

    if new_width is None:
        ratio = new_height / height
        new_width = int(width * ratio)
    elif new_height is None:
        ratio = new_width / width
        new_height = int(height * ratio)

    x_indices = (np.arange(new_width) * (width / new_width)).astype(int)
    y_indices = (np.arange(new_height) * (height / new_height)).astype(int)

    x_indices = np.clip(x_indices, 0, width - 1)
    y_indices = np.clip(y_indices, 0, height - 1)

    resized_image = image[y_indices[:, None], x_indices[None, :]]

    return resized_image


def apply_sepia(image):
    sepia_filter = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ])

    sepia_image = np.dot(image.astype(float), sepia_filter.T)
    sepia_image = np.clip(sepia_image, 0, 255)

    return sepia_image.astype(np.uint8)


def apply_vignette(image, strength=0.8):
    height, width = image.shape[:2]

    x = np.arange(width)
    y = np.arange(height)
    X, Y = np.meshgrid(x, y)

    center_x, center_y = width // 2, height // 2
    max_dist = np.sqrt(center_x ** 2 + center_y ** 2)
    dist = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)

    vignette_factor = 1 - (dist / max_dist) * strength
    vignette_factor = np.clip(vignette_factor, 0, 1)
    vignette_image = image.astype(float) * vignette_factor[:, :, np.newaxis]

    return vignette_image.astype(np.uint8)


def pixelate_region(image, x, y, width, height, pixel_size=10):
    """
    Основная функция пикселизации - отдельно от коллбэков
    Использует ТВОЙ алгоритм с усреднением блоков
    """
    pixelated_image = image.copy()

    # Ограничиваем координаты размерами изображения
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(image.shape[1], x + width)
    y2 = min(image.shape[0], y + height)

    # Пикселизируем область - ТВОЙ АЛГОРИТМ
    for block_y in range(y1, y2, pixel_size):
        for block_x in range(x1, x2, pixel_size):
            block_end_y = min(block_y + pixel_size, y2)
            block_end_x = min(block_x + pixel_size, x2)

            # Вычисляем средний цвет блока
            block = image[block_y:block_end_y, block_x:block_end_x]
            if block.size > 0:
                avg_color = np.mean(block, axis=(0, 1))

                # Заполняем блок средним цветом
                pixelated_image[block_y:block_end_y, block_x:block_end_x] = avg_color

    return pixelated_image


def interactive_pixelation(image, pixel_size=10):
  
    global selection_start, selection_end, current_image

    current_image = image.copy()
    temp_image = image.copy()

    selection_start = None
    selection_end = None

    window_name = "Select Region to Pixelate (Drag mouse, press SPACE to confirm, ESC to cancel)"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        display_image = temp_image.copy()

        # Рисуем прямоугольник выбора
        if selection_start and selection_end:
            x1, y1 = selection_start
            x2, y2 = selection_end
            cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow(window_name, display_image)
        key = cv2.waitKey(1) & 0xFF

        if key == 32:  # SPACE - подтвердить
            if selection_start and selection_end:
                x1, y1 = selection_start
                x2, y2 = selection_end

                x_min, x_max = min(x1, x2), max(x1, x2)
                y_min, y_max = min(y1, y2), max(y1, y2)

                width = x_max - x_min
                height = y_max - y_min

                temp_image = pixelate_region(image, x_min, y_min, width, height, pixel_size)
                current_image = temp_image.copy()
                break

        elif key == 27:  # ESC - отмена
            temp_image = current_image.copy()
            break

    cv2.destroyWindow(window_name)
    return current_image


def add_solid_border(image, border_width=10, border_color=(255, 255, 255)):
    height, width = image.shape[:2]

    bordered_image = np.zeros(
        (height + 2 * border_width, width + 2 * border_width, image.shape[2]),
        dtype=image.dtype
    )

    bordered_image[:, :] = border_color

    bordered_image[border_width:border_width + height,
    border_width:border_width + width] = image

    return bordered_image


def add_patterned_border(image, border_width=20, border_color=(255, 255, 255), pattern_type="zigzag"):
    height, width = image.shape[:2]

    bordered_image = np.zeros(
        (height + 2 * border_width, width + 2 * border_width, image.shape[2]),
        dtype=image.dtype
    )

    bordered_image[border_width:border_width + height,
    border_width:border_width + width] = image
    full_height, full_width = bordered_image.shape[:2]
    Y, X = np.ogrid[:full_height, :full_width]

    if pattern_type == "zigzag":
        # Зигзагообразный узор
        mask = ((X + Y) % 10 < 5)
        border_mask = (
                (X < border_width) | (X >= full_width - border_width) |
                (Y < border_width) | (Y >= full_height - border_width)
        )
        bordered_image[mask & border_mask] = border_color

    elif pattern_type == "dots":
        # Точечный узор
        mask = ((X % 10 == 0) & (Y % 10 == 0))
        border_mask = (
                (X < border_width) | (X >= full_width - border_width) |
                (Y < border_width) | (Y >= full_height - border_width)
        )
        bordered_image[mask & border_mask] = border_color

    elif pattern_type == "waves":
        # Волнообразный узор
        wave_mask = (np.sin(X * 0.1) + np.sin(Y * 0.1)) > 0
        border_mask = (
                (X < border_width) | (X >= full_width - border_width) |
                (Y < border_width) | (Y >= full_height - border_width)
        )
        bordered_image[wave_mask & border_mask] = border_color

    return bordered_image


def apply_lens_flare(image, intensity=0.3, flare_texture_path='lens_flare.jpg'):
    flare_texture = cv2.imread(flare_texture_path)
    if flare_texture is None:
        print(f"Текстура {flare_texture_path} не найдена, создаем базовую текстуру")
    else:
        flare_texture = cv2.resize(flare_texture, (image.shape[1], image.shape[0]))

    # Конвертируем в float для вычислений
    image_float = image.astype(np.float32)
    flare_float = flare_texture.astype(np.float32)

    # Альфа-смешивание
    result = image_float * (1 - intensity) + flare_float * intensity
    result = np.clip(result, 0, 255)

    return result.astype(np.uint8)


def apply_watercolor_texture(image, texture_intensity=0.3, paper_texture_path='watercolor_paper.jpg'):
    height, width = image.shape[:2]

    paper_texture = cv2.imread(paper_texture_path)
    if paper_texture is None:
        print(f"Текстура {paper_texture_path} не найдена, создаем базовую текстуру")
    else:
        paper_texture = cv2.resize(paper_texture, (width, height))

    texture_gray = np.mean(paper_texture, axis=2)
    texture_mask = 1 - (texture_gray / 255.0)
    texture_mask = texture_mask ** (1 / 0.9)
    texture_mask = texture_mask[:, :, np.newaxis]

    blended = (image.astype(np.float32) * (1 - texture_mask * texture_intensity) +
               paper_texture.astype(np.float32) * (texture_mask * texture_intensity))

    return np.clip(blended, 0, 255).astype(np.uint8)


def demonstrate_filters():
    image = cv2.imread('photo.jpg')
    if image is None:
        print("Ошибка: не удалось загрузить изображение")
        return

    print("Демонстрация фильтров обработки изображений")
    print("=" * 50)

    # Изменение разрешения
    resized = change_resolution(image, new_width=400)
    cv2.imshow('Original', image)
    cv2.imshow('Resized', resized)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()

    # Сепия
    sepia = apply_sepia(resized)
    cv2.imshow('Original', resized)
    cv2.imshow('Sepia Effect', sepia)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()

    # Виньетка
    vignette = apply_vignette(resized, strength=0.6)
    cv2.imshow('Original', resized)
    cv2.imshow('Vignette Effect', vignette)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()

    # Пикселизация
    print("Выберите область для пикселизации")
    pixelated = interactive_pixelation(resized)
    cv2.imshow('Pixelated Result', pixelated)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()

    # Рамки
    bordered = add_solid_border(resized, border_width=15, border_color=(0, 0, 255))
    cv2.imshow('Original', resized)
    cv2.imshow('Solid Border', bordered)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()

    patterned_border = add_patterned_border(resized, border_width=25,
                                            border_color=(0, 255, 0),
                                            pattern_type="zigzag")
    cv2.imshow('Original', resized)
    cv2.imshow('Patterned Border', patterned_border)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()

    # Блики
    flare = apply_lens_flare(resized, intensity=0.2)
    cv2.imshow('Original', resized)
    cv2.imshow('Lens Flare', flare)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()

    # Акварельная текстура
    watercolor = apply_watercolor_texture(resized, texture_intensity=0.4)
    cv2.imshow('Original', resized)
    cv2.imshow('Watercolor Texture', watercolor)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Image Processing Filters')
    parser.add_argument('image_path', help='Path to input image')
    parser.add_argument('--filter', required=True,
                        choices=['resize', 'sepia', 'vignette', 'pixelate',
                                 'solid_border', 'patterned_border', 'lens_flare', 'watercolor'],
                        help='Filter to apply')
    parser.add_argument('--width', type=int, help='New width for resize')
    parser.add_argument('--height', type=int, help='New height for resize')
    parser.add_argument('--strength', type=float, default=0.8, help='Strength for vignette')
    parser.add_argument('--border_width', type=int, default=10, help='Border width')
    parser.add_argument('--border_color', nargs=3, type=int, default=[255, 255, 255],
                        help='Border color as B G R')
    parser.add_argument('--pattern', default='zigzag',
                        choices=['zigzag', 'dots', 'waves'],
                        help='Pattern type for patterned border')
    parser.add_argument('--intensity', type=float, default=0.3, help='Intensity for effects')

    args = parser.parse_args()

    image = cv2.imread(args.image_path)
    if image is None:
        print(f"Error: Could not load image from {args.image_path}")
        return

    # Применяем выбранный фильтр
    if args.filter == 'resize':
        result = change_resolution(image, args.width, args.height)
    elif args.filter == 'sepia':
        result = apply_sepia(image)
    elif args.filter == 'vignette':
        result = apply_vignette(image, args.strength)
    elif args.filter == 'pixelate':
        result = interactive_pixelation(image)
    elif args.filter == 'solid_border':
        result = add_solid_border(image, args.border_width, tuple(args.border_color))
    elif args.filter == 'patterned_border':
        result = add_patterned_border(image, args.border_width,
                                      tuple(args.border_color), args.pattern)
    elif args.filter == 'lens_flare':
        result = apply_lens_flare(image, args.intensity)
    elif args.filter == 'watercolor':
        result = apply_watercolor_texture(image, args.intensity)

    # Показываем результаты
    cv2.imshow('Original', image)
    cv2.imshow('Result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":

    if len(sys.argv) > 1:
        main()
    else:
        demonstrate_filters()
