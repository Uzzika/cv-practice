import argparse
import sys
import cv2 as cv
import numpy as np
import os


def resize_image(image, width, height):
    """
    Функция изменения разрешения изображения.
    
    Args:
        image: исходное изображение
        width: новая ширина
        height: новая высота
    
    Returns:
        измененное изображение
    """
    h, w = image.shape[:2]
    
    # Какие СТРОКИ мы буедм брать из исходного изображения
    src_y = ((np.arange(height) / height) * h).astype(int)
    # Какие СТОЛБЦЫ
    src_x = ((np.arange(width) / width) * w).astype(int)
    
    # Создаем сетки координат. По сути из src_x,y делаем двумерную сетку координат
    Y, X = np.meshgrid(src_y, src_x, indexing='ij')
    
    # Создаем новое изображение
    resized = image[Y, X]
    
    return resized


def sepia_filter(image):
    """
    Функция применения фотоэффекта сепии к изображению.
    
    Args:
        image: исходное изображение
    
    Returns:
        изображение с эффектом сепии
    """
    # Матрица преобразования для сепии
    sepia_matrix = np.array([
        [0.272, 0.349, 0.393],
        [0.534, 0.686, 0.769],
        [0.131, 0.168, 0.189]
    ])
    
    # Применяем матрицу преобразования
    sepia = np.clip(image @ sepia_matrix, 0, 255)
    
    return sepia.astype(np.uint8)


def vignette_filter(image, strength=0.5):
    """
    Функция применения фотоэффекта виньетки к изображению.
    
    Args:
        image: исходное изображение
        strength: сила эффекта виньетки (0-1)
    
    Returns:
        изображение с эффектом виньетки
    """
    h, w = image.shape[:2]
    center_x, center_y = w // 2, h // 2
    
    # Создаем координатные сетки
    y, x = np.ogrid[:h, :w]
    # от каждого пикселя расстояние до середины
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # Вычисляем максимальное расстояние до углов
    max_distance = np.sqrt(center_x**2 + center_y**2)
    
    # Создаем маску виньетки. 1 - (distance / max_distance) * strength - это формула для виньетки
    vignette_mask = 1 - (distance / max_distance) * strength
    # Нормализация 0-1 + добавление третьего канала
    vignette_mask = np.clip(vignette_mask, 0, 1)[:, :, np.newaxis]
    
    # Применяем маску к изображению
    return (image * vignette_mask).astype(np.uint8)


def pixelate_region(image, x, y, width, height, pixel_size=10):
    """
    Функция пикселизации заданной прямоугольной области изображения.
    
    Args:
        image: исходное изображение
        x, y: координаты левого верхнего угла области
        width, height: размеры области
        pixel_size: размер пикселя для пикселизации
    
    Returns:
        изображение с пикселизированной областью
    """
    pixelated = image.copy()
    
    # Ограничение координат границами изображения
    x = max(0, min(x, image.shape[1]))
    y = max(0, min(y, image.shape[0]))
    # Проверка выхода размера областиза рамки изображения
    width = min(width, image.shape[1] - x)
    height = min(height, image.shape[0] - y)
    
    # Извлекаем область для пикселизации
    region = image[y:y+height, x:x+width]
    
    # Высчитываем новое разрешение области
    small_width = max(1, width // pixel_size)
    small_height = max(1, height // pixel_size)
    # Не изобретаем велосипед
    small_region = resize_image(region, small_width, small_height)
    
    # Возвращаем к исходному размеру
    pixelated_region = resize_image(small_region, width, height)
    
    # Заменяем область в исходном изображении
    pixelated[y:y+height, x:x+width] = pixelated_region
    
    return pixelated


def interactive_pixelate(image, pixel_size=15):
    """
    Интерактивная пикселизация области с помощью мыши (с коллбэками).
    
    Args:
        image: исходное изображение
        pixel_size: размер пикселя для пикселизации
    
    Returns:
        изображение с пикселизированной областью
    """
    rect_start = None
    rect_end = None
    drawing = False
    result_image = image.copy()
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal rect_start, rect_end, drawing, result_image
        
        if event == cv.EVENT_LBUTTONDOWN:
            drawing = True
            rect_start = (x, y)
            rect_end = (x, y)
        
        elif event == cv.EVENT_MOUSEMOVE:
            if drawing:
                rect_end = (x, y)
        
        elif event == cv.EVENT_LBUTTONUP:
            drawing = False
            rect_end = (x, y)
            
            # Вычисляем параметры для пикселизации
            x1 = min(rect_start[0], rect_end[0])
            y1 = min(rect_start[1], rect_end[1])
            x2 = max(rect_start[0], rect_end[0])
            y2 = max(rect_start[1], rect_end[1])
            width = x2 - x1
            height = y2 - y1
            
            # Применяем пикселизацию
            if width > 0 and height > 0:
                result_image = pixelate_region(image, x1, y1, width, height, pixel_size)
    
    print("   >> Выберите область для пикселизации мышью. Нажмите ESC для завершения...")
    
    window_name = 'Выберите область (нажмите ESC)'
    cv.namedWindow(window_name)
    cv.setMouseCallback(window_name, mouse_callback)
    
    while True:
        display_image = result_image.copy()
        
        # Рисуем прямоугольник выделения
        if rect_start and rect_end:
            cv.rectangle(display_image, rect_start, rect_end, (0, 255, 0), 2)
        
        cv.imshow(window_name, display_image)
        
        key = cv.waitKey(1) & 0xFF
        if key == 27:  # ескейп
            break
    
    cv.destroyAllWindows()
    return result_image


def add_rectangular_frame(image, frame_width, color=(0, 0, 0)):
    """
    Функция наложения прямоугольной одноцветной рамки заданной ширины по краям изображения.
    
    Args:
        image: исходное изображение
        frame_width: ширина рамки
        color: цвет рамки (BGR)
    
    Returns:
        изображение с рамкой
    """
    framed = image.copy()
    
    # Все строки до индекса frame_width и все столбцы.
    framed[:frame_width, :] = color
    # Низ
    framed[-frame_width:, :] = color
    
    # Левая и правая
    framed[:, :frame_width] = color
    framed[:, -frame_width:] = color
    
    return framed


def add_decorative_frame(image, frame_type="wavy", frame_width=20, color=(0, 0, 0)):
    """
    Функция наложения фигурной одноцветной рамки по краям изображения.
    
    Args:
        image: исходное изображение
        frame_type: тип рамки ("wavy", "pattern", "straight", "red", "floral")
        frame_width: ширина рамки
        color: цвет рамки (BGR)
    
    Returns:
        изображение с фигурной рамкой
    """
    # Загружаем текстуру рамки с альфа-каналом
    frame_types = ["wavy", "pattern", "straight", "red", "floral"]
    frame_index = frame_types.index(frame_type) if frame_type in frame_types else 0
    
    frame_path = f"textures/frame{frame_index}.jpg"
    frame = cv.imread(frame_path, cv.IMREAD_UNCHANGED)
    
    h, w = image.shape[:2]
    
    # Изменяем размер рамки под размер изображения
    frame = resize_image(frame, w, h)
    
    # Проверяем наличие альфа-канала
    if len(frame.shape) == 3 and frame.shape[2] == 4:
        # Используем альфа-канал для смешивания
        alpha = frame[:, :, 3:4] / 255.0
        frame_rgb = frame[:, :, :3]
        result = image * (1 - alpha) + frame_rgb * alpha
    else:
        # Без альфа-канала - старый способ (маска)
        # Вычисляем насколько пиксель отличается от белого
        color_diff = np.sqrt(np.sum((frame.astype(np.float32) - 255) ** 2, axis=2))
        # Если отличается больше чем на 30, то этот пиксель будет белым (False) + 3 канал
        frame_mask = (color_diff > 30)[:, :, np.newaxis]
        
        # Где рамка, то ставим рамку, где нет, то ставим исходное изображение
        result = image * ~frame_mask + frame * frame_mask
    
    return result.astype(np.uint8)


def add_lens_flare(image, flare_x, flare_y, intensity=0.8):
    """
    Функция наложения эффекта бликов объектива камеры.
    
    Args:
        image: исходное изображение
        flare_x, flare_y: координаты центра блика
        intensity: интенсивность блика (0-1)
    
    Returns:
        изображение с эффектом блика
    """
    # Загружаем текстуру блика с альфа-каналом
    glare = cv.imread("textures/glare.jpg", cv.IMREAD_UNCHANGED)
    
    h_img, w_img = image.shape[:2]
    
    # Изменяем размер блика
    glare = resize_image(glare, w_img // 4, h_img // 4)
    h_glare, w_glare = glare.shape[:2]
    
    # Вычисляем координаты размещения блика
    y_start = flare_y - h_glare
    y_end = y_start + h_glare
    x_start = flare_x - w_glare
    x_end = x_start + w_glare
    
    # Ограничение на выход за границы изображения
    img_y_start = max(0, y_start)
    img_y_end = min(h_img, y_end)
    img_x_start = max(0, x_start)
    img_x_end = min(w_img, x_end)
    
    # Вычисляем координаты блика на текстуре
    glare_y_start = max(0, -y_start)
    glare_y_end = h_glare - max(0, y_end - h_img)
    glare_x_start = max(0, -x_start)
    glare_x_end = w_glare - max(0, x_end - w_img)
    
    # Применяем блик к изображению
    flared = image.astype(np.float32).copy()
    glare_region = glare[glare_y_start:glare_y_end, glare_x_start:glare_x_end]
    
    # Проверяем наличие альфа-канала
    if glare_region.shape[2] == 4:
        # Используем альфа-канал для смешивания
        alpha = (glare_region[:, :, 3:4] / 255.0) * intensity
        glare_rgb = glare_region[:, :, :3]
        flared[img_y_start:img_y_end, img_x_start:img_x_end] = \
            flared[img_y_start:img_y_end, img_x_start:img_x_end] * (1 - alpha) + glare_rgb * alpha
    else:
        # Без альфа-канала - старый способ
        flared[img_y_start:img_y_end, img_x_start:img_x_end] += glare_region * intensity
    
    # Нормализуем + приведение типа
    return np.clip(flared, 0, 255).astype(np.uint8)


def add_watercolor_texture(image, texture_strength=0.3):
    """
    Функция наложения текстуры акварельной бумаги.
    
    Args:
        image: исходное изображение
        texture_strength: сила текстуры (0-1)
    
    Returns:
        изображение с текстурой акварельной бумаги
    """
    # Загружаем текстуру акварельной бумаги с альфа-каналом
    texture = cv.imread("textures/watercolor_paper.jpg", cv.IMREAD_UNCHANGED)
    
    h, w = image.shape[:2]
    
    # Изменяем размер текстуры под размер изображения
    texture = resize_image(texture, w, h)
    
    # Проверяем наличие альфа-канала
    if len(texture.shape) == 3 and texture.shape[2] == 4:
        # Используем альфа-канал для смешивания
        alpha = (texture[:, :, 3:4] / 255.0) * texture_strength
        texture_rgb = texture[:, :, :3]
        blended = image * (1 - alpha) + texture_rgb * alpha
    else:
        # Без альфа-канала - старый способ
        if len(texture.shape) == 2:
            texture = cv.cvtColor(texture, cv.COLOR_GRAY2BGR)
        
        # Среднее по трём каналам. Как яркость текстуры
        texture_gray = np.mean(texture, axis=2)
        # Нормализация 0-1 + инверсия + степень силы текстуры
        texture_mask = (1 - texture_gray / 255.0) ** (1 / texture_strength)
        texture_mask = texture_mask[:, :, np.newaxis]
        
        # Применяем текстуру к изображению
        blended = image * (1 - texture_mask * texture_strength) + texture * (texture_mask * texture_strength)
    
    return blended.astype(np.uint8)


def cli_argument_parser():
    """Парсер аргументов командной строки."""
    parser = argparse.ArgumentParser(description='Библиотека фильтров изображений')
    
    parser.add_argument('-i', '--image', 
                       help='Путь к изображению', 
                       type=str, 
                       dest='image_path',
                       required=True)
    
    parser.add_argument('-f', '--filter',
                       help='Тип фильтра (resize, sepia, vignette, pixelate, frame, decorative_frame, flare, watercolor)',
                       type=str,
                       dest='filter_type',
                       required=True)
    
    parser.add_argument('-o', '--output',
                       help='Выходное изображение',
                       type=str,
                       dest='output_image',
                       default='output.jpg')
    
    parser.add_argument('--width', type=int, default=800, help='Ширина для resize')
    parser.add_argument('--height', type=int, default=600, help='Высота для resize')
    parser.add_argument('--strength', type=float, default=0.5, help='Сила эффекта')
    parser.add_argument('--x', type=int, default=100, help='X координата для pixelate')
    parser.add_argument('--y', type=int, default=100, help='Y координата для pixelate')
    parser.add_argument('--pixel_size', type=int, default=10, help='Размер пикселя')
    parser.add_argument('--frame_width', type=int, default=20, help='Ширина рамки')
    parser.add_argument('--frame_type', type=str, default='circle', help='Тип декоративной рамки')
    parser.add_argument('--flare_x', type=int, default=400, help='X координата блика')
    parser.add_argument('--flare_y', type=int, default=300, help='Y координата блика')
    
    return parser.parse_args()


def main():
    """Основная функция."""
    args = cli_argument_parser()
    
    # Загружаем изображение
    image = cv.imread(args.image_path)
    if image is None:
        print(f"Ошибка: не удалось загрузить изображение {args.image_path}")
        return 1
    
    print(f"Загружено изображение: {image.shape}")
    
    # Применяем выбранный фильтр
    if args.filter_type == 'resize':
        result = resize_image(image, args.width, args.height)
        print(f"Изменено разрешение на {args.width}x{args.height}")
        
    elif args.filter_type == 'sepia':
        result = sepia_filter(image)
        print("Применен эффект сепии")
        
    elif args.filter_type == 'vignette':
        result = vignette_filter(image, args.strength)
        print(f"Применен эффект виньетки (сила: {args.strength})")
        
    elif args.filter_type == 'pixelate':
        result = pixelate_region(image, args.x, args.y, 200, 200, args.pixel_size)
        print(f"Применена пикселизация области ({args.x}, {args.y})")
        
    elif args.filter_type == 'frame':
        result = add_rectangular_frame(image, args.frame_width)
        print(f"Добавлена прямоугольная рамка (ширина: {args.frame_width})")
        
    elif args.filter_type == 'decorative_frame':
        result = add_decorative_frame(image, args.frame_type, args.frame_width)
        print(f"Добавлена декоративная рамка (тип: {args.frame_type})")
        
    elif args.filter_type == 'flare':
        result = add_lens_flare(image, args.flare_x, args.flare_y, args.strength)
        print(f"Добавлен эффект блика в точке ({args.flare_x}, {args.flare_y})")
        
    elif args.filter_type == 'watercolor':
        result = add_watercolor_texture(image, args.strength)
        print(f"Добавлена текстура акварельной бумаги (сила: {args.strength})")
        
    else:
        print(f"Неизвестный тип фильтра: {args.filter_type}")
        return 1
    
    # Отображаем исходное и обработанное изображения
    cv.imshow('Исходное изображение', image)
    cv.imshow('Обработанное изображение', result)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    # Сохраняем результат
    cv.imwrite(args.output_image, result)
    print(f"Результат сохранен в {args.output_image}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
