import cv2
import numpy as np
import os
import glob


class OpenCVWindowManager:
    def __init__(self, window_name):
        self.window_name = window_name
        self.window = window_name
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        self.current_image = None
        self.original_image = None
        self.background_color = (0, 0, 0)
        self.last_window_size = None
        self.image_files = []
        self.current_image_index = 0

        cv2.setMouseCallback(self.window, self.on_mouse_event)

    def set_size(self, width, height):
        """Установить размер окна"""
        cv2.resizeWindow(self.window, width, height)

    def get_size(self):
        """Получить текущий размер окна"""
        props = cv2.getWindowImageRect(self.window)
        if props is not None:
            x, y, width, height = props
            return width, height
        return None

    def get_position(self):
        """Получить позицию окна"""
        props = cv2.getWindowImageRect(self.window)
        if props is not None:
            x, y, width, height = props
            return x, y
        return None

    def display_image(self, image):
        """Отобразить изображение в окне"""
        self.current_image = image.copy()
        cv2.imshow(self.window, image)

    def wait_key(self, delay=0):
        """Ожидать нажатия клавиши с автоматической проверкой размера окна"""
        key = cv2.waitKey(delay) & 0xFF
        self.check_window_resize()

        return key

    def check_window_resize(self):
        """Проверить и обработать изменение размера окна"""
        current_size = self.get_size()
        if current_size and current_size != self.last_window_size:
            self.update_displayed_image()
            self.last_window_size = current_size

    def on_mouse_event(self, event, x, y, flags, param):
        """Обработчик событий мыши для отслеживания изменения размера"""
        if event in [cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN]:
            self.update_displayed_image()

    def update_displayed_image(self):
        """Обновить отображение изображения при изменении размера окна"""
        if self.original_image is not None:
            self.add_image_with_padding(self.original_image, self.background_color)

    def add_image_with_padding(self, image, background_color=(0, 0, 0)):
        """
        Добавить изображение с сохранением пропорций и черными полосами
        """
        if image is None:
            print("Ошибка: изображение не загружено")
            return

        self.original_image = image.copy()
        self.background_color = background_color

        window_size = self.get_size()

        self.last_window_size = window_size
        window_width, window_height = window_size
        img_height, img_width = image.shape[:2]

        window_ratio = window_width / window_height
        img_ratio = img_width / img_height

        if img_ratio > window_ratio:
            new_width = window_width
            new_height = int(window_width / img_ratio)
        else:
            new_height = window_height
            new_width = int(window_height * img_ratio)

        resized_img = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        padded_img = np.full((window_height, window_width, 3), background_color, dtype=np.uint8)

        y_offset = (window_height - new_height) // 2
        x_offset = (window_width - new_width) // 2

        padded_img[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_img

        self.display_image(padded_img)

        return padded_img

    def load_images_from_folder(self, folder_path="images"):
        """
        Загрузить все изображения из указанной папки
        """
        if not os.path.exists(folder_path):
            print(f"Ошибка: папка '{folder_path}' не существует!")
            return False

        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']

        self.image_files = []

        for extension in extensions:
            self.image_files.extend(glob.glob(os.path.join(folder_path, extension)))

        self.image_files.sort()

        image = self.load_current_image()
        self.add_image_with_padding(image)

        return True

    def load_current_image(self):
        """Загрузить текущее изображение"""

        if self.current_image_index >= len(self.image_files):
            self.current_image_index = 0

        try:
            image = cv2.imread(self.image_files[self.current_image_index])
            if image is None:
                print(f"Ошибка загрузки изображения: {self.image_files[self.current_image_index]}")
                return None
            return image
        except Exception as e:
            print(f"Ошибка при загрузке изображения: {e}")
            return None

    def show_next_image(self):
        """Показать следующее изображение"""
        if not self.image_files:
            print("Нет загруженных изображений!")
            return False

        self.current_image_index = (self.current_image_index + 1) % len(self.image_files)
        image = self.load_current_image()

        if image is not None:
            print(f"\nТекущее изображение: {self.current_image_index + 1}/{len(self.image_files)}")
            self.add_image_with_padding(image)
            return True

        return False

    def show_previous_image(self):
        """Показать предыдущее изображение"""
        if not self.image_files:
            print("Нет загруженных изображений!")
            return False

        self.current_image_index = (self.current_image_index - 1) % len(self.image_files)
        image = self.load_current_image()

        if image is not None:
            print(f"\nТекущее изображение: {self.current_image_index + 1}/{len(self.image_files)}")
            self.add_image_with_padding(image)
            return True

        return False

