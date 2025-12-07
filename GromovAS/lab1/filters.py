import cv2
import numpy as np
import sys

# Фильтры

# Глобальные переменные для пикселизации
pixelate_start = None
pixelate_end = None
drawing = False
original_image = None
current_image = None
pixel_size = 10  # Размер пикселя по умолчанию


# Коллбэк для мыши
def mouse_callback_pixelate(event, x, y, flags, param):
    global pixelate_start, pixelate_end, drawing, original_image, current_image, pixel_size

    if event == cv2.EVENT_LBUTTONDOWN:
        pixelate_start = (x, y)
        pixelate_end = (x, y)
        drawing = True

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        pixelate_end = (x, y)
        # Показываем предварительный просмотр прямоугольника
        temp_image = current_image.copy()
        cv2.rectangle(temp_image, pixelate_start, pixelate_end, (0, 255, 0), 2)
        cv2.imshow("Pixelate Tool - Select Region", temp_image)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        pixelate_end = (x, y)

        # Применяем пикселизацию к выбранной области
        x1, y1 = pixelate_start
        x2, y2 = pixelate_end
        x = min(x1, x2)
        y = min(y1, y2)
        w = abs(x2 - x1)
        h = abs(y2 - y1)

        if w > 0 and h > 0:
            pixelated = pixelate_region(current_image, x, y, w, h, pixel_size)
            current_image = pixelated
            cv2.imshow("Pixelate Tool - Select Region", current_image)

        # Сбрасываем координаты
        pixelate_start = None
        pixelate_end = None


# Функция пикселизации
def pixelate_region(image, x, y, w, h, pixel_size=10):
    img_copy = image.copy()
    x_end = min(x + w, image.shape[1])
    y_end = min(y + h, image.shape[0])

    for i in range(y, y_end, pixel_size):
        for j in range(x, x_end, pixel_size):
            i_end = min(i + pixel_size, y_end)
            j_end = min(j + pixel_size, x_end)
            block = img_copy[i:i_end, j:j_end]
            if block.size > 0:
                avg_color = np.mean(block, axis=(0, 1)).astype(np.uint8)
                img_copy[i:i_end, j:j_end] = avg_color
    return img_copy


# Запуск интерактивного режима пикселизации
def start_pixelate_interactive(image, initial_pixel_size=10):
    """Запускает интерактивный режим пикселизации"""
    global original_image, current_image, pixel_size
    original_image = image.copy()
    current_image = image.copy()
    pixel_size = initial_pixel_size

    cv2.namedWindow("Pixelate Tool - Select Region")
    cv2.setMouseCallback("Pixelate Tool - Select Region", mouse_callback_pixelate)
    cv2.imshow("Pixelate Tool - Select Region", image)

    print("=== Pixelate Tool ===")
    print("Инструкции:")
    print("- Зажмите ЛКМ и выделите область для пикселизации")
    print("- Отпустите ЛКМ для применения эффекта")
    print("- '+'/-': Увеличить/уменьшить размер пикселя (текущий: {})".format(pixel_size))
    print("- 'r': Сбросить к исходному изображению")
    print("- 'q' или ESC: Выйти и вернуть результат")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # Q или ESC
            break
        elif key == ord('r'):
            current_image = original_image.copy()
            cv2.imshow("Pixelate Tool - Select Region", current_image)
            print("Изображение сброшено")
        elif key == ord('+') or key == ord('='):
            pixel_size = min(50, pixel_size + 2)
            print("Размер пикселя увеличен до:", pixel_size)
        elif key == ord('-'):
            pixel_size = max(2, pixel_size - 2)
            print("Размер пикселя уменьшен до:", pixel_size)

    cv2.destroyWindow("Pixelate Tool - Select Region")
    return current_image


# Функция пикселизации всей картинки
def pixelate_whole_image(image, pixel_size=10):
    return pixelate_region(image, 0, 0, image.shape[1], image.shape[0], pixel_size)

# Изменение размера 
def resize_image(img, new_h=None, new_w=None, scale=None):
    h, w = img.shape[:2]

    if scale is not None:
        if scale == 1:
            return img.copy()
        new_h = int(h * scale)
        new_w = int(w * scale)
    elif new_h is None and new_w is None:
        return img.copy()
    elif new_h is None:
        new_h = int(h * new_w / w)
    elif new_w is None:
        new_w = int(w * new_h / h)

    scale_h = h / new_h
    scale_w = w / new_w

    y = (np.arange(new_h) * scale_h).astype(np.int32)
    x = (np.arange(new_w) * scale_w).astype(np.int32)
    x_neigh_index, y_neigh_index = np.meshgrid(x, y)

    res = img[y_neigh_index, x_neigh_index]
    return res

# Сепия
def sepia_effect(image):
    sepia_matrix = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]])
    img_float = image.astype(np.float32)
    sepia_img = np.dot(img_float.reshape(-1, 3), sepia_matrix.T).reshape(img_float.shape)
    return np.clip(sepia_img, 0, 255).astype(np.uint8)

# Виньетка
def vignette_effect(image, strength=0.5):
    h, w = image.shape[:2]
    Y, X = np.ogrid[:h, :w]
    mask = np.exp(-(((X - w/2)**2)/(2*(w*strength)**2) + ((Y - h/2)**2)/(2*(h*strength)**2)))
    mask = mask / mask.max()
    vignette = image.astype(np.float32)
    for c in range(3):
        vignette[:,:,c] *= mask
    return np.clip(vignette, 0, 255).astype(np.uint8)

# Прямоугольная граница
def add_rectangular_border(image, color=(0,255,0), thickness=30):
    bordered = image.copy()
    bordered[:thickness, :] = color
    bordered[-thickness:, :] = color
    bordered[:, :thickness] = color
    bordered[:, -thickness:] = color
    return bordered

# Фигурная граница
def add_shape_border(image, color=(255,0,0), thickness=25, shape_type='ellipse'):
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cy, cx = h//2, w//2
    if shape_type == 'ellipse':
        for i in range(h):
            for j in range(w):
                y_norm = (i - cy)/(h/2 - thickness)
                x_norm = (j - cx)/(w/2 - thickness)
                if x_norm**2 + y_norm**2 >= 1:
                    mask[i,j] = 1
    elif shape_type == 'circle':
        radius = min(h,w)//2 - thickness
        for i in range(h):
            for j in range(w):
                if (i-cy)**2 + (j-cx)**2 >= radius**2:
                    mask[i,j] = 1
    elif shape_type == 'diamond':
        for i in range(h):
            for j in range(w):
                if abs(i-cy) + abs(j-cx) >= min(h,w)//2 - thickness:
                    mask[i,j] = 1
    bordered = image.copy()
    for c in range(3):
        bordered[:,:,c] = np.where(mask==1, color[c], bordered[:,:,c])
    return bordered

# Линза
# Блик через стороннюю текстуру
def lens_flare_texture(image, flare_image_path=None, flare_center=(0.5, 0.5), intensity=1.0, blend_mode='add'):
    h, w = image.shape[:2]
    cx, cy = int(flare_center[0] * w), int(flare_center[1] * h)

    if flare_image_path is None:
        return lens_flare_standard(image, flare_center, intensity)

    try:
        flare_img = cv2.imread(flare_image_path, cv2.IMREAD_UNCHANGED)
        if flare_img is None:
            print(f"Ошибка: не удалось загрузить текстуру блика '{flare_image_path}'")
            return lens_flare_standard(image, flare_center, intensity)

        # Масштабируем текстуру блика
        flare_scale = min(w, h) / 3 / max(flare_img.shape[0], flare_img.shape[1])
        new_flare_w = int(flare_img.shape[1] * flare_scale)
        new_flare_h = int(flare_img.shape[0] * flare_scale)
        flare_resized = cv2.resize(flare_img, (new_flare_w, new_flare_h))

        # Позиционируем текстуру
        flare_x = cx - new_flare_w // 2
        flare_y = cy - new_flare_h // 2

        # Создаем результат
        result = image.astype(np.float32)

        # Обрабатываем в зависимости от количества каналов в текстуре
        if flare_resized.shape[2] == 4:  # RGBA
            flare_rgb = flare_resized[:, :, :3].astype(np.float32)
            flare_alpha = flare_resized[:, :, 3].astype(np.float32) / 255.0

            # Применяем альфа-канал и интенсивность
            flare_effect = flare_rgb * flare_alpha[:, :, np.newaxis] * intensity

        else:
            flare_effect = flare_resized.astype(np.float32) * intensity

        # Накладываем эффект на основное изображение
        for i in range(new_flare_h):
            for j in range(new_flare_w):
                img_i = flare_y + i
                img_j = flare_x + j

                # Проверяем границы
                if 0 <= img_i < h and 0 <= img_j < w:
                    if blend_mode == 'add':
                        result[img_i, img_j] += flare_effect[i, j]
                    elif blend_mode == 'screen':
                        a = result[img_i, img_j] / 255.0
                        b = flare_effect[i, j] / 255.0
                        result[img_i, img_j] = (1 - (1 - a) * (1 - b)) * 255
                    elif blend_mode == 'lighten':
                        result[img_i, img_j] = np.maximum(result[img_i, img_j], flare_effect[i, j])

        return np.clip(result, 0, 255).astype(np.uint8)

    except Exception as e:
        print(f"Ошибка при обработке текстуры блика: {e}")
        return lens_flare_standard(image, flare_center, intensity)


# Стандартный блик (резервный вариант)
def lens_flare_standard(image, flare_center=(0.5, 0.5), intensity=1.0):
    h, w = image.shape[:2]
    cx, cy = int(flare_center[0] * w), int(flare_center[1] * h)

    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

    main_flare = np.exp(-(dist / (0.15 * max(h, w))) ** 2)

    flare1_center = (int(cx * 0.7), int(cy * 0.7))
    flare2_center = (int(cx * 1.3), int(cy * 1.3))
    dist1 = np.sqrt((X - flare1_center[0]) ** 2 + (Y - flare1_center[1]) ** 2)
    dist2 = np.sqrt((X - flare2_center[0]) ** 2 + (Y - flare2_center[1]) ** 2)

    flare1 = np.exp(-(dist1 / (0.08 * max(h, w))) ** 2) * 0.6
    flare2 = np.exp(-(dist2 / (0.05 * max(h, w))) ** 2) * 0.4

    combined_flare = main_flare + flare1 + flare2
    combined_flare = combined_flare / combined_flare.max() * intensity

    flare_texture = np.stack([combined_flare * 1.0, combined_flare * 0.8, combined_flare * 0.6], axis=2)

    result = image.astype(np.float32) + flare_texture * 255
    return np.clip(result, 0, 255).astype(np.uint8)

# Бумага
def watercolor_paper_texture(image, paper_texture_path):
    h, w = image.shape[:2]
    try:
        # Загружаем текстуру бумаги
        texture = cv2.imread(paper_texture_path)
        if texture is None:
            raise ValueError(f"Не удалось загрузить текстуру бумаги '{paper_texture_path}'")
        # Масштабируем текстуру под размер изображения
        scaled_texture = resize_image(texture, h, w)
        # Преобразуем в float и нормализуем
        tex = scaled_texture.astype(np.float32) / 255.0
        tex = tex - tex.mean()  # Вычитаем среднее значение
        img_f = image.astype(np.float32) / 255.0
        k = 0.5
        res = np.clip(img_f + k * tex, 0, 1)
        return (res * 255).astype(np.uint8)

    except Exception as e:
        print(f"Ошибка в watercolor_paper_texture: {e}")
        return image.copy()

# Функция вызова фильтра
def apply_filter(image, filter_type, *args):
    if filter_type == 'resize':
        if len(args) == 1:
            scale = float(args[0])
            return resize_image(image, scale=scale)
        elif len(args) == 2:
            try:
                new_h = int(args[0])
                new_w = int(args[1])
                return resize_image(image, new_h=new_h, new_w=new_w)
            except:
                scale_x = float(args[0])
                scale_y = float(args[1])
                return resize_image(image, scale=scale_x)
        else:
            return resize_image(image, scale=1.0)

    elif filter_type == 'sepia':
        return sepia_effect(image)

    elif filter_type == 'vignette':
        strength = float(args[0]) if len(args) > 0 else 0.5
        return vignette_effect(image, strength)

    elif filter_type == 'pixelate':
        if len(args) == 0:
            # Интерактивный режим
            return start_pixelate_interactive(image)
        elif len(args) == 1:
            # Пикселизация всей картинки
            pixel_size = int(args[0])
            return pixelate_whole_image(image, pixel_size)
        elif len(args) >= 4:
            # Пикселизация конкретной области
            x, y, w, h = map(int, args[:4])
            pixel_size = int(args[4]) if len(args) > 4 else 10
            return pixelate_region(image, x, y, w, h, pixel_size)
        else:
            # Если неправильное количество параметров, запускаем интерактивный режим
            print("Неверное количество параметров. Запуск интерактивного режима...")
            return start_pixelate_interactive(image)

    elif filter_type == 'rect_border':
        thickness = int(args[0]) if len(args) > 0 else 20
        color = tuple(map(int, args[1:4])) if len(args) >= 4 else (0, 255, 0)
        return add_rectangular_border(image, color, thickness)

    elif filter_type == 'shape_border':
        thickness = int(args[0]) if len(args) > 0 else 20
        color = tuple(map(int, args[1:4])) if len(args) >= 4 else (255, 0, 0)
        shape_type = args[4] if len(args) > 4 else 'ellipse'
        return add_shape_border(image, color, thickness, shape_type)

    elif filter_type == 'lens_flare':
        if len(args) == 0:
            return lens_flare_texture(image)
        elif len(args) >= 1:
            try:
                float(args[0])
                flare_center = tuple(map(float, args[:2])) if len(args) >= 2 else (0.5, 0.5)
                intensity = float(args[2]) if len(args) > 2 else 1.0
                return lens_flare_standard(image, flare_center, intensity)
            except:
                texture_path = args[0]
                flare_center = tuple(map(float, args[1:3])) if len(args) >= 3 else (0.5, 0.5)
                intensity = float(args[3]) if len(args) > 3 else 1.0
                blend_mode = args[4] if len(args) > 4 else 'add'
                return lens_flare_texture(image, texture_path, flare_center, intensity, blend_mode)


    elif filter_type == 'watercolor':
        if len(args) == 0:
            raise ValueError("Для watercolor требуется путь к текстуре бумаги")
        texture_path = args[0]
        return watercolor_paper_texture(image, texture_path)
    else:
        raise ValueError("Неизвестный фильтр")

# Main
def main():
    if len(sys.argv) < 3:
        print("Использование: python filters.py <image_path> <filter_type> [filter_params...]")
        print("Доступные фильтры:")
        print("  resize [scale] - изменение размера (масштаб)")
        print("  resize [height] [width] - изменение размера (точные размеры)")
        print("  sepia - сепия")
        print("  vignette [strength] - виньетка")
        print("  pixelate - интерактивная пикселизация мышью")
        print("  pixelate [size] - пикселизация всей картинки")
        print("  pixelate [x y w h] [size] - пикселизация области")
        print("  rect_border [thickness] [r g b] - прямоугольная граница")
        print("  shape_border [thickness] [r g b] [type] - фигурная граница")
        print("  lens_flare [center_x center_y] [intensity] - блик")
        print("  watercolor [intensity] [scale] - акварельная бумага")
        print("\nПримеры для pixelate:")
        print("  python filters.py image.jpg pixelate                   # интерактивный режим")
        print("  python filters.py image.jpg pixelate 15                # вся картинка с размером пикселя 15")
        print("  python filters.py image.jpg pixelate 100 50 200 150    # область 100,50,200,150")
        print("  python filters.py image.jpg pixelate 100 50 200 150 12 # область с размером пикселя 12")
        return

    image_path = sys.argv[1]
    filter_type = sys.argv[2]
    filter_params = sys.argv[3:]

    image = cv2.imread(image_path)
    if image is None:
        print("Ошибка: изображение не найдено")
        return
    try:
        filtered = apply_filter(image, filter_type, *filter_params)
        if filter_type != 'pixelate' or len(filter_params) > 0:
            cv2.imshow("Original", image)
            cv2.imshow(f"Filtered: {filter_type}", filtered)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    except Exception as e:
        print("Ошибка при применении фильтра:", e)
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()
