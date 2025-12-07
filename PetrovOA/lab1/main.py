"""
Практическая работа №1. Обработка изображений с использованием библиотеки OpenCV
Библиотека фильтров для обработки изображений
"""

import cv2
import numpy as np
import argparse
import sys


# ============================================================================
# ФИЛЬТРЫ
# ============================================================================

def resize_image(image, width=None, height=None, scale_factor=None):
    """
    Изменение разрешения изображения
    
    Args:
        image: Входное изображение
        width: Новая ширина
        height: Новая высота
        scale_factor: Коэффициент масштабирования
    
    Returns:
        Изображение с измененным разрешением
    """
    h, w = image.shape[:2]
    
    # Определяем новые размеры
    if scale_factor is not None:
        new_width = max(int(w * scale_factor), 1)
        new_height = max(int(h * scale_factor), 1)
    elif width is not None and height is not None:
        new_width = width
        new_height = height
    elif width is not None:
        ratio = width / w
        new_width = width
        new_height = max(int(h * ratio), 1)
    elif height is not None:
        ratio = height / h
        new_width = max(int(w * ratio), 1)
        new_height = height
    else:
        return image.copy()
    
    # Выводим информацию о разрешениях
    print(f"Изменение разрешения: {w}x{h} -> {new_width}x{new_height}")
    
    # Создаем массив для нового изображения
    resized = np.zeros((new_height, new_width, 3), dtype=image.dtype)
    
    # Создаем координатные сетки для исходного изображения
    # y_ratio и x_ratio показывают, какой пиксель исходного изображения соответствует каждому пикселю нового
    y_ratio = np.linspace(0, h - 1, new_height).astype(int)
    x_ratio = np.linspace(0, w - 1, new_width).astype(int)
    
    # Создаем 2D сетки координат
    # y_indices[i, j] = y_ratio[i] - индекс строки в исходном изображении
    # x_indices[i, j] = x_ratio[j] - индекс столбца в исходном изображении
    y_indices, x_indices = np.meshgrid(y_ratio, x_ratio, indexing='ij')
    
    # Копируем пиксели из исходного изображения в новое
    resized = image[y_indices, x_indices]
    
    return resized


def apply_sepia(image, intensity=1.0):
    """
    Применение фотоэффекта сепии к изображению
    
    Args:
        image: входное изображение
        intensity: интенсивность эффекта (0.0 - 1.0)
    
    Returns:
        изображение с эффектом сепии
    """
    # Матрица преобразования для сепии (BGR формат в OpenCV)
    # Преобразуем BGR -> RGB для применения стандартной матрицы, затем обратно
    sepia_matrix = np.array([
        [0.131, 0.534, 0.272],  # B
        [0.168, 0.686, 0.349],  # G
        [0.189, 0.769, 0.393]   # R
    ])
    
    # Применяем матричное преобразование
    result = image.astype(np.float32)
    sepia_image = result @ sepia_matrix.T
    
    # Обрезаем значения до [0, 255]
    sepia_image = np.clip(sepia_image, 0, 255)
    
    # Смешиваем с оригиналом согласно интенсивности
    result = result * (1 - intensity) + sepia_image * intensity
    
    return result.astype(np.uint8)


def apply_vignette(image, intensity=0.5):
    """
    Применение фотоэффекта виньетки к изображению
    
    Args:
        image: входное изображение
        intensity: интенсивность виньетки (0.0 - 1.0)
    
    Returns:
        изображение с эффектом виньетки
    """
    h, w = image.shape[:2]
    
    # Центр изображения
    center_x, center_y = w / 2, h / 2
    max_distance = np.sqrt(center_x**2 + center_y**2)
    
    # Создаем сетки координат
    y_coords, x_coords = np.ogrid[:h, :w]
    
    # Вычисляем расстояние от центра для всех пикселей сразу
    distance = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
    
    # Нормализуем расстояние
    norm_distance = distance / max_distance
    
    # Вычисляем коэффициент затемнения
    vignette_factor = 1.0 - (norm_distance * intensity)
    vignette_factor = np.maximum(0.0, vignette_factor)
    
    # Добавляем размерность для каналов
    vignette_factor = vignette_factor[:, :, np.newaxis]
    
    # Применяем виньетку
    result = image.astype(np.float32) * vignette_factor
    
    return result.astype(np.uint8)


def pixelate_region(image, x1, y1, x2, y2, pixel_size=10):
    """
    Пикселизация заданной прямоугольной области изображения
    
    Args:
        image: входное изображение
        x1, y1: координаты левого верхнего угла
        x2, y2: координаты правого нижнего угла
        pixel_size: размер пикселя
    
    Returns:
        изображение с пикселизированной областью
    """
    result = np.copy(image)
    
    # Убедимся, что координаты в правильном порядке
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    
    # Ограничение координат границами изображения
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    # Извлекаем область для пикселизации (создаем копию!)
    region = image[y1:y2, x1:x2].copy()
    region_h, region_w = region.shape[:2]
    
    # Вычисляем количество блоков
    n_blocks_y = (region_h + pixel_size - 1) // pixel_size
    n_blocks_x = (region_w + pixel_size - 1) // pixel_size
    
    # Создаем пикселизированную версию
    for i in range(n_blocks_y):
        for j in range(n_blocks_x):
            # Границы текущего блока
            block_y1 = i * pixel_size
            block_y2 = min(block_y1 + pixel_size, region_h)
            block_x1 = j * pixel_size
            block_x2 = min(block_x1 + pixel_size, region_w)
            
            # Вычисляем средний цвет блока векторно
            block = region[block_y1:block_y2, block_x1:block_x2]
            avg_color = np.mean(block, axis=(0, 1))
            
            # Заполняем блок средним цветом
            region[block_y1:block_y2, block_x1:block_x2] = avg_color
    
    # Вставляем обработанную область обратно
    result[y1:y2, x1:x2] = region
    
    return result


def add_simple_frame(image, frame_width=20, color=(0, 0, 0)):
    """
    Наложение прямоугольной одноцветной рамки заданной ширины по краям изображения
    
    Args:
        image: входное изображение
        frame_width: ширина рамки в пикселях
        color: цвет рамки в формате BGR
    
    Returns:
        изображение с рамкой
    """
    result = np.copy(image)
    h, w = result.shape[:2]
    
    # Верхняя рамка
    result[0:frame_width, :] = color
    
    # Нижняя рамка
    result[h-frame_width:h, :] = color
    
    # Левая рамка
    result[:, 0:frame_width] = color
    
    # Правая рамка
    result[:, w-frame_width:w] = color
    
    return result


def add_decorative_frame(image, frame_width=20, color=(0, 0, 0), frame_type='rounded'):
    """
    Наложение фигурной одноцветной рамки по краям изображения
    
    Args:
        image: входное изображение
        frame_width: ширина рамки в пикселях
        color: цвет рамки в формате BGR
        frame_type: тип рамки ('rounded', 'wave', 'zigzag')
    
    Returns:
        изображение с фигурной рамкой
    """
    result = np.copy(image)
    h, w = result.shape[:2]
    
    # Создаем сетки координат
    y_coords, x_coords = np.ogrid[:h, :w]
    
    if frame_type == 'rounded':
        # Скругленные углы
        radius = frame_width
        
        # Маски для углов
        in_top_left = ((y_coords < radius) & (x_coords < radius) & 
                      ((y_coords - radius)**2 + (x_coords - radius)**2 > radius**2))
        in_top_right = ((y_coords < radius) & (x_coords >= w - radius) & 
                       ((y_coords - radius)**2 + (x_coords - w + radius)**2 > radius**2))
        in_bottom_left = ((y_coords >= h - radius) & (x_coords < radius) & 
                         ((y_coords - h + radius)**2 + (x_coords - radius)**2 > radius**2))
        in_bottom_right = ((y_coords >= h - radius) & (x_coords >= w - radius) & 
                          ((y_coords - h + radius)**2 + (x_coords - w + radius)**2 > radius**2))
        
        # Маска для прямых краев
        on_edge = ((y_coords < frame_width) | (y_coords >= h - frame_width) | 
                  (x_coords < frame_width) | (x_coords >= w - frame_width))
        
        # Общая маска рамки
        mask = on_edge | in_top_left | in_top_right | in_bottom_left | in_bottom_right
        result[mask] = color
    
    elif frame_type == 'wave':
        # Волнообразная рамка
        amplitude = frame_width // 3
        frequency = 0.05
        
        # Вычисляем волны для всех координат сразу
        wave_offset_h = (amplitude * np.sin(x_coords * frequency)).astype(np.int32)
        wave_offset_v = (amplitude * np.sin(y_coords * frequency)).astype(np.int32)
        
        # Границы с учетом волн
        wave_top = frame_width + wave_offset_h
        wave_bottom = h - frame_width - wave_offset_h
        wave_left = frame_width + wave_offset_v
        wave_right = w - frame_width - wave_offset_v
        
        # Маска рамки
        mask = ((y_coords < wave_top) | (y_coords >= wave_bottom) | 
                (x_coords < wave_left) | (x_coords >= wave_right))
        result[mask] = color
    
    elif frame_type == 'zigzag':
        # Зигзагообразная рамка
        zigzag_size = max(1, frame_width // 2)
        
        # Вычисляем зигзаг-смещения
        zigzag_h = ((y_coords // zigzag_size) % 2) * zigzag_size
        zigzag_v = ((x_coords // zigzag_size) % 2) * zigzag_size
        
        # Маска рамки
        on_top = y_coords < frame_width + zigzag_v
        on_bottom = y_coords >= h - frame_width - zigzag_v
        on_left = x_coords < frame_width + zigzag_h
        on_right = x_coords >= w - frame_width - zigzag_h
        
        mask = on_top | on_bottom | on_left | on_right
        result[mask] = color
    
    return result


def add_lens_flare(image, center_x=None, center_y=None, intensity=0.8):
    """
    Наложение эффекта бликов объектива камеры с использованием готовой текстуры
    
    Args:
        image: входное изображение
        center_x: не используется (для совместимости)
        center_y: не используется (для совместимости)
        intensity: интенсивность блика (0.0 - 1.0)
    
    Returns:
        изображение с эффектом бликов
    """
    import os
    
    h, w = image.shape[:2]
    
    # Загружаем текстуру блика с альфа-каналом
    script_dir = os.path.dirname(os.path.abspath(__file__))
    blik_path = os.path.join(script_dir, 'blik.png')
    
    if not os.path.exists(blik_path):
        print(f"Предупреждение: текстура блика не найдена: {blik_path}")
        return image
    
    blik = cv2.imread(blik_path, cv2.IMREAD_UNCHANGED)
    if blik is None:
        print(f"Предупреждение: не удалось загрузить текстуру блика: {blik_path}")
        return image
    
    # Подгоняем размер текстуры под размер изображения
    blik = cv2.resize(blik, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # Проверяем наличие альфа-канала
    if blik.shape[2] == 4:
        # Извлекаем альфа-канал и нормализуем
        alpha = blik[:, :, 3] / 255.0
        blik_rgb = blik[:, :, :3].astype(np.float32)
        img_f = image.astype(np.float32)
        
        # Смешиваем изображения с учетом альфа-канала
        for c in range(3):
            img_f[:, :, c] = img_f[:, :, c] * (1 - alpha * intensity) + blik_rgb[:, :, c] * (alpha * intensity)
        
        result = np.clip(img_f, 0, 255).astype(np.uint8)
    else:
        # Если альфа-канала нет, используем простое смешивание
        blik_rgb = blik.astype(np.float32)
        img_f = image.astype(np.float32)
        result = img_f * (1 - intensity) + blik_rgb * intensity
        result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result


def add_watercolor_texture(image, intensity=0.3):
    """
    Наложение текстуры акварельной бумаги из файла
    
    Args:
        image: входное изображение
        intensity: интенсивность текстуры (0.0 - 1.0)
    
    Returns:
        изображение с текстурой акварельной бумаги
    """
    import os
    
    # Загружаем текстуру акварельной бумаги
    script_dir = os.path.dirname(os.path.abspath(__file__))
    paper_path = os.path.join(script_dir, 'paper.png')
    
    if not os.path.exists(paper_path):
        print(f"Предупреждение: текстура бумаги не найдена: {paper_path}")
        return image
    
    paper_texture = cv2.imread(paper_path)
    if paper_texture is None:
        print(f"Предупреждение: не удалось загрузить текстуру бумаги: {paper_path}")
        return image
    
    h, w = image.shape[:2]
    
    # Подгоняем размер текстуры под размер изображения
    paper_resized = cv2.resize(paper_texture, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # Конвертируем изображения в float32 для точных вычислений
    result = image.astype(np.float32)
    paper_float = paper_resized.astype(np.float32)
    
    # Складываем изображение с текстурой с учетом интенсивности
    result = result + paper_float * intensity
    
    # Обрезаем значения до диапазона [0, 255]
    result = np.clip(result, 0, 255)
    
    return result.astype(np.uint8)


# ============================================================================
# ИНТЕРАКТИВНЫЙ ИНТЕРФЕЙС
# ============================================================================

class ImageFilterApp:
    def __init__(self, image_path, filter_type, **filter_params):
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")
        
        self.filter_type = filter_type
        self.filter_params = filter_params
        self.display_image = np.copy(self.original_image)
        
        # Параметры для интерактивной пикселизации
        self.pixelate_mode = (filter_type == 'pixelate')
        self.drawing = False
        self.ix, self.iy = -1, -1
        self.fx, self.fy = -1, -1
        
        self.window_name = 'Image Filter Demo'
    
    def mouse_callback(self, event, x, y, flags, param):
        """Обработчик событий мыши для интерактивной пикселизации"""
        if not self.pixelate_mode:
            return
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y
            self.display_image = np.copy(self.original_image)
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                temp_image = np.copy(self.display_image)
                # Рисуем прямоугольник для визуализации выделения
                cv2.rectangle(temp_image, (self.ix, self.iy), (x, y), (0, 255, 0), 2)
                cv2.imshow(self.window_name, temp_image)
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.fx, self.fy = x, y
            
            # Применяем пикселизацию к выделенной области
            pixel_size = self.filter_params.get('pixel_size', 10)
            self.display_image = pixelate_region(
                self.display_image, 
                self.ix, self.iy, 
                self.fx, self.fy, 
                pixel_size
            )
            cv2.imshow(self.window_name, self.display_image)
    
    def apply_filter(self):
        """Применение выбранного фильтра"""
        if self.filter_type == 'resize':
            width = self.filter_params.get('width')
            height = self.filter_params.get('height')
            scale_factor = self.filter_params.get('scale')
            return resize_image(self.original_image, width, height, scale_factor)
        
        elif self.filter_type == 'sepia':
            intensity = self.filter_params.get('intensity', 1.0)
            return apply_sepia(self.original_image, intensity)
        
        elif self.filter_type == 'vignette':
            intensity = self.filter_params.get('intensity', 0.5)
            return apply_vignette(self.original_image, intensity)
        
        elif self.filter_type == 'pixelate':
            # Интерактивный режим - возвращаем оригинал
            return np.copy(self.original_image)
        
        elif self.filter_type == 'simple_frame':
            frame_width = self.filter_params.get('frame_width', 20)
            color = self.filter_params.get('color', (0, 0, 0))
            return add_simple_frame(self.original_image, frame_width, color)
        
        elif self.filter_type == 'decorative_frame':
            frame_width = self.filter_params.get('frame_width', 20)
            color = self.filter_params.get('color', (0, 0, 0))
            frame_style = self.filter_params.get('frame_style', 'rounded')
            return add_decorative_frame(self.original_image, frame_width, color, frame_style)
        
        elif self.filter_type == 'lens_flare':
            center_x = self.filter_params.get('center_x')
            center_y = self.filter_params.get('center_y')
            intensity = self.filter_params.get('intensity', 0.8)
            return add_lens_flare(self.original_image, center_x, center_y, intensity)
        
        elif self.filter_type == 'watercolor':
            intensity = self.filter_params.get('intensity', 0.3)
            return add_watercolor_texture(self.original_image, intensity)
        
        else:
            print(f"Неизвестный тип фильтра: {self.filter_type}")
            return np.copy(self.original_image)
    
    def run(self):
        """Запуск приложения"""
        # Применяем фильтр
        filtered_image = self.apply_filter()
        self.display_image = filtered_image
        
        # Создаем окно
        cv2.namedWindow(self.window_name)
        
        # Для интерактивной пикселизации устанавливаем callback
        if self.pixelate_mode:
            cv2.setMouseCallback(self.window_name, self.mouse_callback)
            print("Интерактивный режим пикселизации: выделите область мышью")
        
        # Показываем изображения
        if self.pixelate_mode:
            cv2.imshow(self.window_name, self.display_image)
        else:
            # Для других фильтров показываем оригинал и результат
            h1, w1 = self.original_image.shape[:2]
            h2, w2 = filtered_image.shape[:2]
            
            # Приводим к общей высоте для корректного объединения
            target_height = max(h1, h2)
            
            # Масштабируем оригинал и результат к одной высоте
            if h1 != target_height:
                scale_1 = target_height / h1
                display_original = cv2.resize(self.original_image, None, fx=scale_1, fy=scale_1)
            else:
                display_original = self.original_image.copy()
            
            if h2 != target_height:
                scale_2 = target_height / h2
                display_filtered = cv2.resize(filtered_image, None, fx=scale_2, fy=scale_2)
            else:
                display_filtered = filtered_image.copy()
            
            # Дополнительное масштабирование если изображения слишком большие
            max_display_height = 800
            if target_height > max_display_height:
                scale_factor = max_display_height / target_height
                display_original = cv2.resize(display_original, None, fx=scale_factor, fy=scale_factor)
                display_filtered = cv2.resize(display_filtered, None, fx=scale_factor, fy=scale_factor)
            
            # Объединяем изображения для сравнения
            combined = np.hstack([display_original, display_filtered])
            cv2.imshow(self.window_name, combined)
        
        print("Нажмите любую клавишу для выхода...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Библиотека фильтров для обработки изображений'
    )
    
    parser.add_argument('image', help='Путь к изображению для обработки')
    parser.add_argument('filter', 
                       choices=['resize', 'sepia', 'vignette', 'pixelate', 
                               'simple_frame', 'decorative_frame', 'lens_flare', 'watercolor'],
                       help='Тип фильтра')
    
    # Параметры для resize
    parser.add_argument('--width', type=int, help='Новая ширина изображения')
    parser.add_argument('--height', type=int, help='Новая высота изображения')
    parser.add_argument('--scale', type=float, help='Коэффициент масштабирования')
    
    # Параметры для эффектов с интенсивностью
    parser.add_argument('--intensity', type=float, default=0.8, 
                       help='Интенсивность эффекта (0.0 - 1.0)')
    
    # Параметры для пикселизации
    parser.add_argument('--pixel-size', type=int, default=10, 
                       help='Размер пикселя для пикселизации')
    
    # Параметры для рамок
    parser.add_argument('--frame-width', type=int, default=20, 
                       help='Ширина рамки в пикселях')
    parser.add_argument('--frame-style', 
                       choices=['rounded', 'wave', 'zigzag'], 
                       default='rounded',
                       help='Стиль фигурной рамки')
    parser.add_argument('--color', nargs=3, type=int, default=[0, 0, 0],
                       help='Цвет рамки в формате B G R')
    
    # Параметры для lens flare
    parser.add_argument('--center-x', type=int, help='X-координата центра блика')
    parser.add_argument('--center-y', type=int, help='Y-координата центра блика')
    
    args = parser.parse_args()
    
    # Подготовка параметров фильтра
    filter_params = {}
    
    if args.filter == 'resize':
        filter_params = {
            'width': args.width,
            'height': args.height,
            'scale': args.scale
        }
    elif args.filter in ['sepia', 'vignette', 'lens_flare', 'watercolor']:
        filter_params['intensity'] = args.intensity
    elif args.filter == 'pixelate':
        filter_params['pixel_size'] = args.pixel_size
    elif args.filter in ['simple_frame', 'decorative_frame']:
        filter_params['frame_width'] = args.frame_width
        filter_params['color'] = tuple(args.color)
        if args.filter == 'decorative_frame':
            filter_params['frame_style'] = args.frame_style
    
    if args.filter == 'lens_flare':
        if args.center_x is not None:
            filter_params['center_x'] = args.center_x
        if args.center_y is not None:
            filter_params['center_y'] = args.center_y
    
    try:
        app = ImageFilterApp(args.image, args.filter, **filter_params)
        app.run()
    except Exception as e:
        print(f"Ошибка: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
