import cv2
import numpy as np
import math
import argparse


# =========================
# 1) Изменение разрешения
# =========================
def change_resolution_nn(image, new_width, new_height):
    """
    Изменение разрешения методом ближайшего соседа (Nearest Neighbor).
    Только базовые операции над матрицами.
    """
    old_h, old_w = image.shape[:2]

    # Защита от некорректных значений
    new_width = max(1, int(new_width))
    new_height = max(1, int(new_height))

    scale_x = old_w / new_width
    scale_y = old_h / new_height

    if image.ndim == 3:
        out = np.zeros((new_height, new_width, image.shape[2]), dtype=image.dtype)
    else:
        out = np.zeros((new_height, new_width), dtype=image.dtype)

    for y in range(new_height):
        src_y = int(y * scale_y)
        if src_y >= old_h:
            src_y = old_h - 1
        for x in range(new_width):
            src_x = int(x * scale_x)
            if src_x >= old_w:
                src_x = old_w - 1
            out[y, x] = image[src_y, src_x]

    return out


# =========================
# 2) Сепия
# =========================
def apply_sepia_bgr(image_bgr):
    """
    Сепия по формуле. Работаем в BGR (как читает OpenCV).
    """
    img = image_bgr.astype(np.float32)

    b = img[:, :, 0]
    g = img[:, :, 1]
    r = img[:, :, 2]

    # Переводим классическую формулу RGB в BGR-раскладку:
    # newR = 0.393R + 0.769G + 0.189B
    # newG = 0.349R + 0.686G + 0.168B
    # newB = 0.272R + 0.534G + 0.131B
    new_r = 0.393 * r + 0.769 * g + 0.189 * b
    new_g = 0.349 * r + 0.686 * g + 0.168 * b
    new_b = 0.272 * r + 0.534 * g + 0.131 * b

    out = np.empty_like(img)
    out[:, :, 2] = np.clip(new_r, 0, 255)
    out[:, :, 1] = np.clip(new_g, 0, 255)
    out[:, :, 0] = np.clip(new_b, 0, 255)

    return out.astype(np.uint8)


# =========================
# 3) Виньетка
# =========================
def apply_vignette(image_bgr, strength=0.8):
    """
    Виньетка: затемнение к краям через маску расстояния до центра.
    strength: 0..1
    """
    strength = float(strength)
    strength = max(0.0, min(1.0, strength))

    h, w = image_bgr.shape[:2]
    img = image_bgr.astype(np.float32)

    # координатные сетки
    x = np.arange(w, dtype=np.float32)
    y = np.arange(h, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(x, y)

    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0

    # нормировка расстояния к [0..1]
    dx = (x_grid - cx) / max(1.0, cx)
    dy = (y_grid - cy) / max(1.0, cy)
    dist = np.sqrt(dx * dx + dy * dy)
    dist = np.clip(dist, 0.0, 1.0)

    mask = 1.0 - dist * strength  # 1 в центре, меньше к краям
    mask = np.clip(mask, 0.0, 1.0)

    if img.ndim == 3:
        for c in range(3):
            img[:, :, c] *= mask
    else:
        img *= mask

    return np.clip(img, 0, 255).astype(np.uint8)


# =========================
# 4) Пикселизация области
# =========================
def apply_pixelation(image_bgr, x, y, width, height, pixel_size=10):
    """
    Пикселизация прямоугольной области (block averaging).
    """
    pixel_size = max(1, int(pixel_size))

    h, w = image_bgr.shape[:2]
    x = int(x)
    y = int(y)
    width = int(width)
    height = int(height)

    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    width = max(1, min(width, w - x))
    height = max(1, min(height, h - y))

    out = image_bgr.copy()

    region = out[y:y + height, x:x + width]

    for i in range(0, height, pixel_size):
        for j in range(0, width, pixel_size):
            bh = min(pixel_size, height - i)
            bw = min(pixel_size, width - j)
            block = region[i:i + bh, j:j + bw]

            # среднее по блоку (BGR)
            if block.ndim == 3:
                avg = np.mean(block, axis=(0, 1))
                region[i:i + bh, j:j + bw] = avg
            else:
                avg = np.mean(block)
                region[i:i + bh, j:j + bw] = avg

    out[y:y + height, x:x + width] = region
    return out


# =========================
# 5) Прямоугольная рамка
# =========================
def add_rectangular_border(image_bgr, border_width=10, border_color=(0, 0, 255)):
    """
    Одноцветная рамка по краям изображения.
    border_color задается в BGR.
    """
    bw = max(1, int(border_width))
    h, w = image_bgr.shape[:2]
    bw = min(bw, min(h // 2, w // 2)) if min(h, w) >= 2 else 1

    out = image_bgr.copy()
    c = np.array(border_color, dtype=out.dtype)

    # верх/низ
    out[0:bw, :] = c
    out[h - bw:h, :] = c
    # лево/право
    out[:, 0:bw] = c
    out[:, w - bw:w] = c

    return out


# =========================
# 6) Фигурная рамка
# =========================
def add_decorative_border(image_bgr, border_width=20, border_color=(0, 255, 255), border_type="wave"):
    """
    Фигурная одноцветная рамка: wave / zigzag / dots / triangles
    border_color в BGR.
    """
    bw = max(1, int(border_width))
    h, w = image_bgr.shape[:2]
    bw = min(bw, min(h // 2, w // 2)) if min(h, w) >= 2 else 1

    out = image_bgr.copy()
    color = np.array(border_color, dtype=out.dtype)

    mask = np.zeros((h, w), dtype=bool)

    if border_type == "wave":
        amplitude = bw * 0.4
        frequency = 2.0 * math.pi / max(1, int(w * 0.25))  # чтобы волна была видимой

        # верх/низ
        for x in range(w):
            top_h = bw - int(amplitude * math.sin(x * frequency))
            bot_h = bw - int(amplitude * math.sin(x * frequency + math.pi))
            top_h = max(1, min(bw, top_h))
            bot_h = max(1, min(bw, bot_h))
            mask[0:top_h, x] = True
            mask[h - bot_h:h, x] = True

        # лево/право
        frequency_y = 2.0 * math.pi / max(1, int(h * 0.25))
        for y in range(h):
            left_w = bw - int(amplitude * math.sin(y * frequency_y))
            right_w = bw - int(amplitude * math.sin(y * frequency_y + math.pi))
            left_w = max(1, min(bw, left_w))
            right_w = max(1, min(bw, right_w))
            mask[y, 0:left_w] = True
            mask[y, w - right_w:w] = True

    elif border_type == "zigzag":
        period = max(2, bw * 2)
        # строим "пилу" для верхней/нижней границы
        for x in range(w):
            t = x % period
            # высота зубца 1..bw
            zig_h = 1 + int((bw - 1) * (t / (period - 1)))
            if (x // period) % 2 == 1:
                zig_h = 1 + (bw - zig_h)
            mask[0:zig_h, x] = True
            mask[h - zig_h:h, x] = True

        for y in range(h):
            t = y % period
            zig_w = 1 + int((bw - 1) * (t / (period - 1)))
            if (y // period) % 2 == 1:
                zig_w = 1 + (bw - zig_w)
            mask[y, 0:zig_w] = True
            mask[y, w - zig_w:w] = True

    elif border_type == "dots":
        spacing = max(2, bw)  # расстояние между точками
        # базовая "обычная" рамка толщиной bw, но красим точками
        # верх/низ
        for yy in range(bw):
            for x in range(0, w, spacing):
                mask[yy, x] = True
                mask[h - 1 - yy, x] = True
        # лево/право
        for xx in range(bw):
            for y in range(0, h, spacing):
                mask[y, xx] = True
                mask[y, w - 1 - xx] = True

    elif border_type == "triangles":
        period = max(2, bw * 2)
        # треугольные "зубцы": высота меняется 1..bw..1
        for x in range(w):
            t = x % period
            if t <= period // 2:
                tri_h = 1 + int((bw - 1) * (t / max(1, period // 2)))
            else:
                tri_h = 1 + int((bw - 1) * ((period - t - 1) / max(1, period // 2)))
            tri_h = max(1, min(bw, tri_h))
            mask[0:tri_h, x] = True
            mask[h - tri_h:h, x] = True

        for y in range(h):
            t = y % period
            if t <= period // 2:
                tri_w = 1 + int((bw - 1) * (t / max(1, period // 2)))
            else:
                tri_w = 1 + int((bw - 1) * ((period - t - 1) / max(1, period // 2)))
            tri_w = max(1, min(bw, tri_w))
            mask[y, 0:tri_w] = True
            mask[y, w - tri_w:w] = True

    else:
        # неизвестный тип — вернем как есть
        return out

    # применяем маску
    out[mask] = color
    return out


# =========================
# 7) Блики объектива
# =========================
def apply_lens_flare(image_bgr, flare_radius=50, intensity=0.7, center_x=None, center_y=None):
    """
    Простой "блик" в виде яркого гауссового пятна.
    """
    flare_radius = max(1, int(flare_radius))
    intensity = float(intensity)
    intensity = max(0.0, min(1.0, intensity))

    out = image_bgr.astype(np.float32)
    h, w = out.shape[:2]

    cx = (w // 2) if center_x is None else int(center_x)
    cy = (h // 2) if center_y is None else int(center_y)
    cx = max(0, min(cx, w - 1))
    cy = max(0, min(cy, h - 1))

    y0 = max(0, cy - flare_radius)
    y1 = min(h, cy + flare_radius + 1)
    x0 = max(0, cx - flare_radius)
    x1 = min(w, cx + flare_radius + 1)

    for y in range(y0, y1):
        dy = y - cy
        for x in range(x0, x1):
            dx = x - cx
            dist = math.sqrt(dx * dx + dy * dy)
            if dist <= flare_radius:
                nd = dist / flare_radius
                # гаусс-подобная яркость
                add = 255.0 * intensity * math.exp(-(nd * nd) * 3.0)
                out[y, x, 0] += add
                out[y, x, 1] += add
                out[y, x, 2] += add

    return np.clip(out, 0, 255).astype(np.uint8)


# =========================
# 8) Текстура акварельной бумаги
# =========================
def apply_watercolor_texture(image_bgr, texture_strength=0.3, seed=1):
    """
    Генерируем "бумажную" текстуру шумом + несколько уровней крупности,
    затем умножаем яркость изображения на (texture/255) и смешиваем.
    Без cv2.resize (используем наш nearest neighbor).
    """
    texture_strength = float(texture_strength)
    texture_strength = max(0.0, min(1.0, texture_strength))

    h, w = image_bgr.shape[:2]
    img = image_bgr.astype(np.float32)

    rng = np.random.default_rng(int(seed))

    # базовый шум
    paper = rng.random((h, w), dtype=np.float32) * 255.0

    # добавляем несколько "крупных" шумов через даунскейл/апскейл nearest neighbor
    for scale in (6, 12, 24):
        sh = max(1, h // scale)
        sw = max(1, w // scale)
        coarse = (rng.random((sh, sw), dtype=np.float32) * 255.0).astype(np.float32)

        # nearest neighbor upsample без cv2.resize:
        coarse_u8 = np.clip(coarse, 0, 255).astype(np.uint8)
        coarse_up = change_resolution_nn(coarse_u8, w, h).astype(np.float32)

        paper = paper * 0.7 + coarse_up * 0.3

    # нормируем к [0..255]
    pmin = float(np.min(paper))
    pmax = float(np.max(paper))
    if pmax > pmin:
        paper = (paper - pmin) / (pmax - pmin) * 255.0
    paper = np.clip(paper, 0, 255)

    # умножение на текстуру + смешивание
    tex = paper / 255.0
    out = img.copy()
    for c in range(3):
        blended = img[:, :, c] * tex
        out[:, :, c] = img[:, :, c] * (1.0 - texture_strength) + blended * texture_strength

    return np.clip(out, 0, 255).astype(np.uint8)


# =========================
# CLI / Демонстрация
# =========================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Практическая работа №1: библиотека фильтров OpenCV (базовые операции над матрицами)"
    )
    parser.add_argument("--image", "-i", required=True, help="Путь к изображению (jpg/png/...)")
    parser.add_argument(
        "--filter", "-f", required=True,
        choices=[
            "resize", "sepia", "vignette", "pixelate",
            "rect_border", "decor_border", "lens_flare", "watercolor", "all"
        ],
        help="Тип фильтра"
    )

    # resize
    parser.add_argument("--new_w", type=int, default=300, help="Новая ширина (для resize)")
    parser.add_argument("--new_h", type=int, default=200, help="Новая высота (для resize)")

    # vignette
    parser.add_argument("--strength", type=float, default=0.8, help="Сила эффекта (0..1)")

    # pixelate
    parser.add_argument("--x", type=int, default=0, help="X (pixelate)")
    parser.add_argument("--y", type=int, default=0, help="Y (pixelate)")
    parser.add_argument("--w", type=int, default=200, help="Width (pixelate)")
    parser.add_argument("--h", type=int, default=200, help="Height (pixelate)")
    parser.add_argument("--pixel_size", type=int, default=15, help="Размер пикселя (pixelate)")

    # borders
    parser.add_argument("--border_width", type=int, default=20, help="Толщина рамки")
    parser.add_argument("--border_b", type=int, default=0, help="B компонента цвета рамки")
    parser.add_argument("--border_g", type=int, default=255, help="G компонента цвета рамки")
    parser.add_argument("--border_r", type=int, default=255, help="R компонента цвета рамки")
    parser.add_argument("--border_type", default="wave", choices=["wave", "zigzag", "dots", "triangles"],
                        help="Тип фигурной рамки (decor_border)")

    # lens flare
    parser.add_argument("--flare_radius", type=int, default=40, help="Радиус блика")
    parser.add_argument("--intensity", type=float, default=0.8, help="Интенсивность блика (0..1)")
    parser.add_argument("--cx", type=int, default=None, help="Центр блика X (опционально)")
    parser.add_argument("--cy", type=int, default=None, help="Центр блика Y (опционально)")

    # watercolor
    parser.add_argument("--texture_strength", type=float, default=0.4, help="Сила текстуры (0..1)")
    parser.add_argument("--seed", type=int, default=1, help="Seed для шума текстуры")

    return parser.parse_args()


def apply_selected_filter(img_bgr, args):
    color = (args.border_b, args.border_g, args.border_r)

    if args.filter == "resize":
        return change_resolution_nn(img_bgr, args.new_w, args.new_h)

    if args.filter == "sepia":
        return apply_sepia_bgr(img_bgr)

    if args.filter == "vignette":
        return apply_vignette(img_bgr, strength=args.strength)

    if args.filter == "pixelate":
        # если x,y по умолчанию 0 — удобнее пикселизовать центр
        if args.x == 0 and args.y == 0:
            h, w = img_bgr.shape[:2]
            x = max(0, w // 2 - args.w // 2)
            y = max(0, h // 2 - args.h // 2)
        else:
            x, y = args.x, args.y
        return apply_pixelation(img_bgr, x, y, args.w, args.h, pixel_size=args.pixel_size)

    if args.filter == "rect_border":
        return add_rectangular_border(img_bgr, border_width=args.border_width, border_color=color)

    if args.filter == "decor_border":
        return add_decorative_border(
            img_bgr, border_width=args.border_width, border_color=color, border_type=args.border_type
        )

    if args.filter == "lens_flare":
        return apply_lens_flare(
            img_bgr, flare_radius=args.flare_radius, intensity=args.intensity,
            center_x=args.cx, center_y=args.cy
        )

    if args.filter == "watercolor":
        return apply_watercolor_texture(
            img_bgr, texture_strength=args.texture_strength, seed=args.seed
        )

    if args.filter == "all":
        # демонстрация всех фильтров: вернем коллаж 2x4 (упрощенно)
        # делаем одинаковый размер миниатюр
        thumb_w, thumb_h = 320, 240
        base = change_resolution_nn(img_bgr, thumb_w, thumb_h)

        items = [
            ("Original", base),
            ("Sepia", change_resolution_nn(apply_sepia_bgr(img_bgr), thumb_w, thumb_h)),
            ("Vignette", change_resolution_nn(apply_vignette(img_bgr, 0.8), thumb_w, thumb_h)),
            ("Pixelate", change_resolution_nn(apply_pixelation(img_bgr, thumb_w//2-60, thumb_h//2-60, 120, 120, 12),
                                             thumb_w, thumb_h)),
            ("RectBorder", change_resolution_nn(add_rectangular_border(img_bgr, 14, (0, 0, 255)), thumb_w, thumb_h)),
            ("WaveBorder", change_resolution_nn(add_decorative_border(img_bgr, 18, (0, 255, 255), "wave"),
                                               thumb_w, thumb_h)),
            ("LensFlare", change_resolution_nn(apply_lens_flare(img_bgr, 35, 0.8), thumb_w, thumb_h)),
            ("Watercolor", change_resolution_nn(apply_watercolor_texture(img_bgr, 0.45, seed=1), thumb_w, thumb_h)),
        ]

        # собираем 2 ряда по 4
        row1 = np.hstack([items[i][1] for i in range(4)])
        row2 = np.hstack([items[i][1] for i in range(4, 8)])
        collage = np.vstack([row1, row2])
        return collage

    # fallback
    return img_bgr


def main():
    args = parse_args()

    img = cv2.imread(args.image)
    if img is None:
        print("Ошибка: не удалось загрузить изображение. Проверьте путь.")
        return

    result = apply_selected_filter(img, args)

    cv2.imshow("Original", img)
    cv2.imshow("Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()