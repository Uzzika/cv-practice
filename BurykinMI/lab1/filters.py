import numpy as np
import cv2


# Изменение разрешения изображения
def resize_image(img, scale=0.5):
    h, w = img.shape[:2]
    new_h, new_w = int(h * scale), int(w * scale)

    # Интерполяция по ближайшему соседу
    # h=100, new_h=50, то y_idx = [0, 2, 4, 6, ..., 98]
    y_idx = (np.linspace(0, h - 1, new_h)).astype(int)
    x_idx = (np.linspace(0, w - 1, new_w)).astype(int)

    resized = img[y_idx][:, x_idx]

    return resized


# Сепия
def apply_sepia(img):
    # BGR матрица
    sepia_filter = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]])
    sepia_img = img @ sepia_filter.T
    sepia_img = np.clip(sepia_img, 0, 255).astype(np.uint8)  # 0-255
    return sepia_img


# Виньетка
def apply_vignette(img, strength=0.5):
    rows, cols = img.shape[:2]
    x = np.linspace(-1, 1, cols)
    y = np.linspace(-1, 1, rows)
    X, Y = np.meshgrid(x, y)  # Двумерные сетки координат, X Y координат пикселей
    radius = np.sqrt(X ** 2 + Y ** 2)
    mask = np.exp(-radius ** 2 / (2 * strength ** 2))
    vignette = img * mask[..., np.newaxis]  # Добавляет новую размерность
    return np.clip(vignette, 0, 255).astype(np.uint8)


# Пикселизация области с выбором мышью
class PixelateSelector:
    def __init__(self, img, pixel_size=10):
        self.img = img.copy()
        self.original = img.copy()
        self.pixel_size = pixel_size
        self.start_point = None
        self.end_point = None
        self.drawing = False

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.img = self.original.copy()

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.img = self.original.copy()
                cv2.rectangle(self.img, self.start_point, (x, y), (0, 255, 0), 2)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end_point = (x, y)
            x1 = min(self.start_point[0], self.end_point[0])
            y1 = min(self.start_point[1], self.end_point[1])
            x2 = max(self.start_point[0], self.end_point[0])
            y2 = max(self.start_point[1], self.end_point[1])

            if x1 < x2 and y1 < y2:
                self.img = pixelate_region(self.original, x1, y1, x2, y2, self.pixel_size)
                self.original = self.img.copy()

    def select_region(self):
        cv2.namedWindow('Pixelate - выберите область')
        cv2.setMouseCallback('Pixelate - выберите область', self.mouse_callback)

        while True:
            cv2.imshow('Pixelate - выберите область', self.img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # q или ESC
                break

        cv2.destroyAllWindows()
        return self.img


# Пикселизация области
def pixelate_region(img, x1, y1, x2, y2, pixel_size=10):
    img_copy = img.copy()
    region = img_copy[y1:y2, x1:x2]  # Вырезка
    h, w = region.shape[:2]
    for y in range(0, h, pixel_size):
        for x in range(0, w, pixel_size):
            block = region[y:y + pixel_size, x:x + pixel_size]
            if block.size > 0:
                color = block.mean(axis=(0, 1))
                region[y:y + pixel_size, x:x + pixel_size] = color
    img_copy[y1:y2, x1:x2] = region
    return img_copy


# Прямоугольная рамка
def add_rectangular_frame(img, color=(0, 0, 255), thickness=20):
    framed = img.copy()
    framed[:thickness, :] = color
    framed[-thickness:, :] = color
    framed[:, :thickness] = color
    framed[:, -thickness:] = color
    return framed


# Фигурная рамка
def add_figure_frame(img, color=(0, 255, 0), thickness=20, frame_type='wave'):
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    x = np.arange(w)
    if frame_type == 'wave':
        # x / 20 длина волны, 10 высота
        y = (np.sin(x / 20) * 10 + thickness).astype(np.int32)
        for i in range(w):
            mask[:y[i], i] = 1
            mask[-y[i]:, i] = 1
    elif frame_type == 'triangle':
        for i in range(thickness):
            mask[i::thickness * 2, :] = 1
            mask[-i - 1::thickness * 2, :] = 1
    mask[:, :thickness] = 1
    mask[:, -thickness:] = 1
    frame = img.copy()
    frame[mask > 0] = color
    return frame


# Эффект бликов через текстуру
def add_lens_flare(img, texture_path, intensity=0.7):
    flare_texture = cv2.imread(texture_path)
    if flare_texture is None:
        print(f"Ошибка: не удалось загрузить текстуру блика из {texture_path}")
        return img

    h, w = img.shape[:2]
    # Изменение размера текстуры
    flare_h, flare_w = flare_texture.shape[:2]
    scale_y = h / flare_h
    scale_x = w / flare_w

    # Интерполяция по ближайшему соседу
    y_idx = (np.linspace(0, flare_h - 1, h)).astype(int)
    x_idx = (np.linspace(0, flare_w - 1, w)).astype(int)
    flare_resized = flare_texture[y_idx][:, x_idx]

    # Нормализация
    img_float = img.astype(np.float32) / 255.0
    flare_float = flare_resized.astype(np.float32) / 255.0

    # Режим наложения "Screen" (осветление)
    result = 1 - (1 - img_float) * (1 - flare_float * intensity)

    return np.clip(result * 255, 0, 255).astype(np.uint8)


# Текстура акварельной бумаги
def add_paper_texture(img, texture_path, intensity=0.3):
    paper_texture = cv2.imread(texture_path)
    if paper_texture is None:
        print(f"Ошибка: не удалось загрузить текстуру бумаги из {texture_path}")
        return img

    h, w = img.shape[:2]
    # Изменение размера текстуры и интерполяция
    paper_h, paper_w = paper_texture.shape[:2]
    y_idx = (np.linspace(0, paper_h - 1, h)).astype(int)
    x_idx = (np.linspace(0, paper_w - 1, w)).astype(int)
    paper_resized = paper_texture[y_idx][:, x_idx]

    # Преобразование в оттенки серого через взвешенное среднее (формула luminosity)
    # Gray = 0.299*R + 0.587*G + 0.114*B (для BGR: 0.114*B + 0.587*G + 0.299*R)
    paper_gray = (paper_resized[:, :, 0] * 0.114 +
                  paper_resized[:, :, 1] * 0.587 +
                  paper_resized[:, :, 2] * 0.299)
    paper_normalized = paper_gray.astype(np.float32) / 255.0

    # Расширение до 3 каналов
    paper_3ch = np.repeat(paper_normalized[:, :, np.newaxis], 3, axis=2)

    # Нормализация
    img_float = img.astype(np.float32) / 255.0

    # Режим наложения "Multiply" (умножение)
    result = img_float * (1 - intensity + paper_3ch * intensity)

    return np.clip(result * 255, 0, 255).astype(np.uint8)