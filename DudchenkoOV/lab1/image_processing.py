import cv2
import numpy as np
import math
import argparse


# =========================
# 1) Масштабирование изображения
# =========================
def change_resolution_nn(image, target_width, target_height):
    """
    Изменение размера методом ближайшего соседа (Nearest Neighbor).
    Используются только базовые операции над массивами.
    """
    src_h, src_w = image.shape[:2]

    # Защита от некорректных входных значений
    target_width = max(1, int(target_width))
    target_height = max(1, int(target_height))

    step_x = src_w / target_width
    step_y = src_h / target_height

    if image.ndim == 3:
        resized = np.zeros((target_height, target_width, image.shape[2]), dtype=image.dtype)
    else:
        resized = np.zeros((target_height, target_width), dtype=image.dtype)

    for row in range(target_height):
        src_row = int(row * step_y)
        if src_row >= src_h:
            src_row = src_h - 1

        for col in range(target_width):
            src_col = int(col * step_x)
            if src_col >= src_w:
                src_col = src_w - 1

            resized[row, col] = image[src_row, src_col]

    return resized


# =========================
# 2) Эффект сепии
# =========================
def apply_sepia_bgr(image_bgr):
    """
    Применение сепии по формуле.
    Изображение обрабатывается в формате BGR, который использует OpenCV.
    """
    float_img = image_bgr.astype(np.float32)

    blue = float_img[:, :, 0]
    green = float_img[:, :, 1]
    red = float_img[:, :, 2]

    # Перевод классической формулы из RGB в порядок каналов BGR:
    # newR = 0.393R + 0.769G + 0.189B
    # newG = 0.349R + 0.686G + 0.168B
    # newB = 0.272R + 0.534G + 0.131B
    sepia_r = 0.393 * red + 0.769 * green + 0.189 * blue
    sepia_g = 0.349 * red + 0.686 * green + 0.168 * blue
    sepia_b = 0.272 * red + 0.534 * green + 0.131 * blue

    sepia_img = np.empty_like(float_img)
    sepia_img[:, :, 2] = np.clip(sepia_r, 0, 255)
    sepia_img[:, :, 1] = np.clip(sepia_g, 0, 255)
    sepia_img[:, :, 0] = np.clip(sepia_b, 0, 255)

    return sepia_img.astype(np.uint8)


# =========================
# 3) Эффект виньетки
# =========================
def apply_vignette(image_bgr, strength=0.8):
    """
    Виньетка: затемнение изображения к краям
    с помощью маски расстояния от центра.
    strength: 0..1
    """
    strength = float(strength)
    strength = max(0.0, min(1.0, strength))

    img_h, img_w = image_bgr.shape[:2]
    float_img = image_bgr.astype(np.float32)

    # Формируем координатные сетки
    x_coords = np.arange(img_w, dtype=np.float32)
    y_coords = np.arange(img_h, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(x_coords, y_coords)

    center_x = (img_w - 1) / 2.0
    center_y = (img_h - 1) / 2.0

    # Нормируем расстояние к диапазону [0..1]
    norm_x = (grid_x - center_x) / max(1.0, center_x)
    norm_y = (grid_y - center_y) / max(1.0, center_y)
    radius = np.sqrt(norm_x * norm_x + norm_y * norm_y)
    radius = np.clip(radius, 0.0, 1.0)

    vignette_mask = 1.0 - radius * strength
    vignette_mask = np.clip(vignette_mask, 0.0, 1.0)

    if float_img.ndim == 3:
        for channel in range(3):
            float_img[:, :, channel] *= vignette_mask
    else:
        float_img *= vignette_mask

    return np.clip(float_img, 0, 255).astype(np.uint8)


# =========================
# 4) Пикселизация области
# =========================
def apply_pixelation(image, area_x=0, area_y=0, area_width=10, area_height=10, pixel_size=10):
    result_img = image.copy()
    img_h, img_w = image.shape[:2]

    pixel_size = max(1, int(pixel_size))

    area_x = max(0, min(int(area_x), img_w - 1))
    area_y = max(0, min(int(area_y), img_h - 1))
    area_width = max(1, min(int(area_width), img_w - area_x))
    area_height = max(1, min(int(area_height), img_h - area_y))

    crop = result_img[area_y:area_y + area_height, area_x:area_x + area_width]

    down_w = max(1, area_width // pixel_size)
    down_h = max(1, area_height // pixel_size)

    reduced_crop = change_resolution_nn(crop, target_width=down_w, target_height=down_h)
    pixelated_crop = change_resolution_nn(reduced_crop, target_width=area_width, target_height=area_height)

    result_img[area_y:area_y + area_height, area_x:area_x + area_width] = pixelated_crop
    return result_img


# =========================
# 5) Прямоугольная рамка
# =========================
def apply_frame(image, border_size=10, border_color=(255, 255, 255)):
    img_h, img_w = image.shape[:2]

    framed = np.full((img_h, img_w, 3), border_color, dtype=np.uint8)
    framed[border_size:img_h - border_size, border_size:img_w - border_size] = \
        image[border_size:img_h - border_size, border_size:img_w - border_size]

    return framed


# =========================
# 6) Декоративная рамка
# =========================
def apply_figure_frame(image, frame_path):
    frame_img = cv2.imread(frame_path)
    if frame_img is None:
        raise ValueError(f"Не удалось загрузить рамку: {frame_path}")

    if frame_img.shape[:2] != image.shape[:2]:
        # Подгоняем рамку под размер исходного изображения
        frame_img = cv2.resize(frame_img, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

    blended = image.copy().astype(np.float32)

    brightness_map = np.mean(frame_img, axis=2) / 255.0
    alpha = (brightness_map <= 0.9).astype(np.float32)

    blended = (
        image.astype(np.float32) * (1 - alpha[:, :, np.newaxis]) +
        frame_img.astype(np.float32) * alpha[:, :, np.newaxis]
    )

    return np.clip(blended, 0, 255).astype(np.uint8)


# =========================
# 7) Блики объектива
# =========================
def apply_lens_flare(image, flare_path, intensity=0.7, position=None):
    flare_img = cv2.imread(flare_path)
    if flare_img is None:
        raise ValueError(f"Не удалось загрузить блик: {flare_path}")

    flare_lightness = np.mean(flare_img, axis=2) / 255.0
    flare_alpha = np.where(flare_lightness > 0.1, flare_lightness * intensity, 0)

    flare_h, flare_w = flare_img.shape[:2]
    img_h, img_w = image.shape[:2]

    if position is None:
        flare_x = max(0, img_w - flare_w - 50)
        flare_y = 50
    else:
        flare_x, flare_y = position

    flare_x = max(0, min(flare_x, img_w - flare_w))
    flare_y = max(0, min(flare_y, img_h - flare_h))

    output = image.copy().astype(np.float32)

    end_y = min(flare_y + flare_h, img_h)
    end_x = min(flare_x + flare_w, img_w)
    overlay_h = end_y - flare_y
    overlay_w = end_x - flare_x

    if overlay_h <= 0 or overlay_w <= 0:
        return image

    flare_part = flare_img[:overlay_h, :overlay_w]
    alpha_part = flare_alpha[:overlay_h, :overlay_w]
    base_part = output[flare_y:end_y, flare_x:end_x]

    flare_float = flare_part.astype(np.float32)

    screen_mix = 255 - (255 - base_part) * (255 - flare_float) / 255

    alpha_3d = alpha_part[:, :, np.newaxis]
    mixed_part = base_part * (1 - alpha_3d) + screen_mix * alpha_3d
    output[flare_y:end_y, flare_x:end_x] = mixed_part

    return np.clip(output, 0, 255).astype(np.uint8)


# =========================
# 8) Текстура акварельной бумаги
# =========================
def watercolor_texture(image, intensity=1.0):
    texture_path = "paper.jpg"
    texture_img = cv2.imread(texture_path)
    if texture_img is None:
        raise ValueError(f"Не удалось загрузить текстуру: {texture_path}")

    if texture_img.shape[:2] != image.shape[:2]:
        texture_img = change_resolution_nn(texture_img, image.shape[1], image.shape[0])

    texture_gray = (
        0.299 * texture_img[:, :, 2] +
        0.587 * texture_img[:, :, 1] +
        0.114 * texture_img[:, :, 0]
    )

    texture_alpha = 1 - (texture_gray / 255.0)
    texture_alpha = texture_alpha[:, :, np.newaxis] * intensity

    mixed = (
        image.astype(np.float32) * (1 - texture_alpha) +
        texture_img.astype(np.float32) * texture_alpha
    )

    return np.clip(mixed, 0, 255).astype(np.uint8)


# =========================
# CLI / Демонстрация
# =========================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Практическая работа №1: библиотека фильтров OpenCV"
    )
    parser.add_argument("--image", "-i", required=True, help="Путь к изображению")
    parser.add_argument(
        "--filter", "-f", required=True,
        choices=[
            "resize", "sepia", "vignette", "pixelate",
            "rect_border", "decor_border", "lens_flare", "watercolor", "all"
        ],
        help="Тип фильтра"
    )

    # Параметры для resize
    parser.add_argument("--new_w", type=int, default=300, help="Новая ширина")
    parser.add_argument("--new_h", type=int, default=200, help="Новая высота")

    # Параметры для vignette
    parser.add_argument("--strength", type=float, default=0.8, help="Сила эффекта")

    # Параметры для pixelate
    parser.add_argument("--x", type=int, default=0, help="X")
    parser.add_argument("--y", type=int, default=0, help="Y")
    parser.add_argument("--w", type=int, default=200, help="Ширина области")
    parser.add_argument("--h", type=int, default=200, help="Высота области")
    parser.add_argument("--pixel_size", type=int, default=15, help="Размер пикселя")

    # Параметры для rect_border
    parser.add_argument("--border_width", type=int, default=20, help="Толщина рамки")
    parser.add_argument("--border_b", type=int, default=255, help="B")
    parser.add_argument("--border_g", type=int, default=255, help="G")
    parser.add_argument("--border_r", type=int, default=255, help="R")

    # Параметры для decor_border
    parser.add_argument(
        "--border_texture",
        type=str,
        default=None,
        help="Путь к изображению для декоративной рамки"
    )

    # Параметры для lens_flare
    parser.add_argument(
        "--lens_flare_texture",
        type=str,
        default=None,
        help="Путь к изображению блика"
    )
    parser.add_argument("--intensity", type=float, default=0.8, help="Интенсивность блика")
    parser.add_argument("--cx", type=int, default=None, help="Позиция блика X")
    parser.add_argument("--cy", type=int, default=None, help="Позиция блика Y")

    # Параметры для watercolor
    parser.add_argument(
        "--texture_strength",
        type=float,
        default=0.4,
        help="Интенсивность текстуры акварели"
    )

    return parser.parse_args()


def apply_selected_filter(image_bgr, args):
    border_color = (args.border_b, args.border_g, args.border_r)

    if args.filter == "resize":
        return change_resolution_nn(image_bgr, args.new_w, args.new_h)

    if args.filter == "sepia":
        return apply_sepia_bgr(image_bgr)

    if args.filter == "vignette":
        return apply_vignette(image_bgr, strength=args.strength)

    if args.filter == "pixelate":
        if args.x == 0 and args.y == 0:
            img_h, img_w = image_bgr.shape[:2]
            area_x = max(0, img_w // 2 - args.w // 2)
            area_y = max(0, img_h // 2 - args.h // 2)
        else:
            area_x, area_y = args.x, args.y

        return apply_pixelation(
            image_bgr,
            area_x,
            area_y,
            args.w,
            args.h,
            pixel_size=args.pixel_size
        )

    if args.filter == "rect_border":
        return apply_frame(image_bgr, border_size=args.border_width, border_color=border_color)

    if args.filter == "decor_border":
        if not args.border_texture:
            print("Ошибка: для decor_border нужно указать --border_texture")
            return image_bgr
        return apply_figure_frame(image_bgr, args.border_texture)

    if args.filter == "lens_flare":
        if not args.lens_flare_texture:
            print("Ошибка: для lens_flare нужно указать --lens_flare_texture")
            return image_bgr

        flare_position = None
        if args.cx is not None and args.cy is not None:
            flare_position = (args.cx, args.cy)

        return apply_lens_flare(
            image_bgr,
            flare_path=args.lens_flare_texture,
            intensity=args.intensity,
            position=flare_position
        )

    if args.filter == "watercolor":
        return watercolor_texture(image_bgr, intensity=args.texture_strength)

    if args.filter == "all":
        preview_w, preview_h = 320, 240
        preview_base = change_resolution_nn(image_bgr, preview_w, preview_h)

        gallery_items = [
            ("Original", preview_base),
            ("Sepia", change_resolution_nn(apply_sepia_bgr(image_bgr), preview_w, preview_h)),
            ("Vignette", change_resolution_nn(apply_vignette(image_bgr, 0.8), preview_w, preview_h)),
            ("Pixelate", change_resolution_nn(
                apply_pixelation(image_bgr, image_bgr.shape[1] // 2 - 60, image_bgr.shape[0] // 2 - 60, 120, 120, 12),
                preview_w, preview_h
            )),
            ("RectBorder", change_resolution_nn(
                apply_frame(image_bgr, border_size=14, border_color=(0, 0, 255)),
                preview_w, preview_h
            )),
            ("DecorBorder", change_resolution_nn(
                apply_figure_frame(image_bgr, args.border_texture) if args.border_texture else image_bgr,
                preview_w, preview_h
            )),
            ("LensFlare", change_resolution_nn(
                apply_lens_flare(image_bgr, args.lens_flare_texture, 0.8) if args.lens_flare_texture else image_bgr,
                preview_w, preview_h
            )),
            ("Watercolor", change_resolution_nn(
                watercolor_texture(image_bgr, intensity=args.texture_strength),
                preview_w, preview_h
            )),
        ]

        top_row = np.hstack([gallery_items[i][1] for i in range(4)])
        bottom_row = np.hstack([gallery_items[i][1] for i in range(4, 8)])
        collage = np.vstack([top_row, bottom_row])
        return collage

    return image_bgr


def main():
    args = parse_args()

    # Загружаем исходное изображение
    source_img = cv2.imread(args.image)
    if source_img is None:
        print("Ошибка: не удалось загрузить изображение. Проверьте путь.")
        return

    # Применяем выбранный фильтр
    filtered_img = apply_selected_filter(source_img, args)

    # Показываем оригинал и результат обработки
    cv2.imshow("Original", source_img)
    cv2.imshow("Result", filtered_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()