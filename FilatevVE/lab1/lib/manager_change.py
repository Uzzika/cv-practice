import os

from lib.change_images import *


def change_sepia(window):
    """
    Менеджер для функции применения фотоэффекта сепии к изображению.
    """
    print("\nФотоэффект сепии:")
    print("Управление интенсивностью с помощью колесика мыши")
    print("Выход из режима - клавиша 'Esc'")

    intensity = 1
    def mouse_callback(event, x, y, flags, param):
        nonlocal intensity
        if event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0 and intensity <= 0.9:
                intensity += 0.05
            elif flags < 0 and intensity >= 0.1:
                intensity -= 0.05

    cv2.setMouseCallback(window.window, mouse_callback)

    while True:
        key = window.wait_key(60)

        if key == 27:
            break

        image = window.load_current_image()
        new_image = sepia_filter(image, intensity)
        window.add_image_with_padding(new_image)

    cv2.setMouseCallback(window.window_name, lambda *args: None)
    window.add_image_with_padding(window.load_current_image())


def change_apply_vignette_elliptical(window):
    """
    Менеджер для функции применения фотоэффекта виньетки к изображению.
    """
    print("\nФотоэффект виньетки:")
    print("Управление размером виньетки с помощью колесика мыши")
    print("Управление интенсивностью с помощью клавиш W и S")
    print("Выход из режима - клавиша 'Esc'")

    current_scale = 1.5
    current_intensity = 0.7

    def mouse_callback(event, x, y, flags, param):
        nonlocal current_scale
        if event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0 and current_scale <= 4.5:
                current_scale += 0.1
            elif flags < 0 and current_scale >= 1.0:
                current_scale -= 0.1

    cv2.setMouseCallback(window.window, mouse_callback)

    while True:
        key = window.wait_key(60)

        if key == ord('w') and current_intensity <= 0.9:
            current_intensity += 0.1

        elif key == ord('s') and current_intensity >= 0.1:
            current_intensity -= 0.1

        elif key == 27:
            break

        image = window.load_current_image()
        new_image = apply_vignette_elliptical(image, current_scale, current_intensity)
        window.add_image_with_padding(new_image)

    cv2.setMouseCallback(window.window_name, lambda *args: None)
    window.add_image_with_padding(window.load_current_image())


def changer_add_rectangular_border(window):
    """
    Менеджер для функции для управления функцией, которая накладывает прямоугольную одноцветную рамку по краям изображения.
    """
    print("\nОдноцветная рамка:")
    print("Управление цветом с помощью колесика мыши")
    print("Выбор цветовой компоненты - нажатие правой кнопки мыши")
    print("Управление толщиной рамки - клавиши W и S")
    print("Выход из режима - клавиша 'Esc'")

    color = [100, 150, 200]
    current_channel = 0
    border_width = 10

    def mouse_callback(event, x, y, flags, param):
        nonlocal color
        nonlocal current_channel
        if event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:
                color[current_channel] = min(color[current_channel] + 5, 255)
            elif flags < 0:
                color[current_channel] = max(color[current_channel] - 5, 0)
        elif event == cv2.EVENT_LBUTTONDOWN:
            current_channel = (current_channel + 1) % 3

    cv2.setMouseCallback(window.window, mouse_callback)

    while True:
        key = window.wait_key(60)

        if key == ord('w'):
            border_width += 1

        elif key == ord('s') and border_width >= 1:
            border_width -= 1

        if key == 27:
            break

        image = window.load_current_image()
        new_image = add_rectangular_border(image, border_width, tuple(color))
        window.add_image_with_padding(new_image)

    cv2.setMouseCallback(window.window_name, lambda *args: None)
    window.add_image_with_padding(window.load_current_image())


def change_add_lens_flare(window):
    """
    Менеджер для функции наложения текстуры блика на изображение.
    """
    print("\nЭффект блика объектива (наложение текстуры):")
    print("Управление интенсивностью - колесико мыши (вверх - увеличить, вниз - уменьшить)")
    print("Позиция блика - клик левой кнопкой мыши")
    print("Управление размером блика - клавиши W (увеличить) и S (уменьшить)")
    print("Выход из режима - клавиша 'Esc'")

    flare_position = (0.5, 0.5)
    intensity = 1.0
    flare_size = 0.3

    def mouse_callback(event, x, y, flags, param):
        nonlocal intensity
        nonlocal flare_position
        if event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0 and intensity <= 0.9:
                intensity += 0.05
            elif flags < 0 and intensity >= 0.1:
                intensity -= 0.05
        elif event == cv2.EVENT_LBUTTONDOWN:
            window_size = window.get_size()
            flare_position = (x / window_size[0], y / window_size[1])

    cv2.setMouseCallback(window.window, mouse_callback)

    while True:
        key = window.wait_key(60)

        if key == ord('w') and flare_size <= 0.9:
            flare_size += 0.1

        elif key == ord('s') and flare_size >= 0.2:
            flare_size -= 0.1

        elif key == 27:
            break

        image = window.load_current_image()
        new_image = add_lens_flare(image, flare_position, intensity, flare_size)
        window.add_image_with_padding(new_image)


    cv2.setMouseCallback(window.window_name, lambda *args: None)
    window.add_image_with_padding(window.load_current_image())


def change_add_watercolor_texture(window):
    """
    Менеджер для функции наложения текстуры акварельной бумаги из файла на изображение.
    """
    print("\nТекстура акварельной бумаги (наложение текстуры):")
    print("Управление интенсивностью текстуры - колесико мыши (вверх - увеличить, вниз - уменьшить)")
    print("Управление силой применения - клавиши W (увеличить) и S (уменьшить)")
    print("Выход из режима - клавиша 'Esc'")

    texture_intensity = 0.3
    strength = 0.9

    def mouse_callback(event, x, y, flags, param):
        nonlocal texture_intensity
        if event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0 and texture_intensity <= 0.9:
                texture_intensity += 0.05
            elif flags < 0 and texture_intensity >= 0.1:
                texture_intensity -= 0.05

    cv2.setMouseCallback(window.window, mouse_callback)

    while True:
        key = window.wait_key(60)

        if key == ord('w') and strength <= 0.95:
            strength += 0.05

        elif key == ord('s') and strength >= 0.05:
            strength -= 0.05

        elif key == 27:
            break

        image = window.load_current_image()
        new_image = add_watercolor_texture(image, texture_intensity, strength)
        window.add_image_with_padding(new_image)

    cv2.setMouseCallback(window.window_name, lambda *args: None)
    window.add_image_with_padding(window.load_current_image())


def figure_frame_manager(window):
    """
    Менеджер для выбора и наложения фигурной рамки
    """
    print("\nРежим фигурной рамки:")
    print("Выбор рамки - клавиши W (следующая) и S (предыдущая)")
    print("Выход из режима - клавиша Esc")

    frames_dir = 'frames'

    frame_files = [f for f in os.listdir(frames_dir)
                   if f.lower().endswith(('.png'))]

    current_frame_index = 0

    while True:
        image = window.load_current_image()

        current_frame_file = frame_files[current_frame_index]
        preview = apply_figure_frame(image, current_frame_file)

        window.add_image_with_padding(preview)

        key = window.wait_key(60)

        if key == 27:
            break
        elif key == ord('w'):
            current_frame_index = (current_frame_index + 1) % len(frame_files)
        elif key == ord('s'):
            current_frame_index = (current_frame_index - 1) % len(frame_files)

    window.add_image_with_padding(window.load_current_image())


def pixelate_region_manager(window):
    """
    Менеджер для пикселизации области
    """
    print("\nРежим пикселизации:")
    print("Выберите область для пикселизации с помощью мыши")
    print("Управление размером пикселя - клавиши W  и S")
    print("Выход из режима - клавиша Esc")

    drawing = False
    ix, iy = -1, -1
    fx, fy = -1, -1
    current_frame_coords = (-1, -1, -1, -1)
    pixel_size = 10

    image = window.load_current_image()
    clone = image.copy()

    def mouse_callback(event, x, y, flags, param):
        nonlocal ix, iy, fx, fy, drawing, current_frame_coords, clone

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
            fx, fy = x, y
            clone = image.copy()

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                fx, fy = x, y
                clone = image.copy()

                x1, y1 = min(ix, fx), min(iy, fy)
                x2, y2 = max(ix, fx), max(iy, fy)

                window_size = window.get_size()
                new_image = draw_green_frame(clone,
                                      (x1 / window_size[0], y1 / window_size[1]),
                                      (x2 / window_size[0], y2 / window_size[1]))

                window.add_image_with_padding(new_image)
                current_frame_coords = (x1, y1, x2, y2)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            fx, fy = x, y

            x1, y1 = min(ix, fx), min(iy, fy)
            x2, y2 = max(ix, fx), max(iy, fy)
            current_frame_coords = (x1, y1, x2, y2)

            clone = image.copy()
            window_size = window.get_size()
            new_image = draw_green_frame(clone,
                                              (x1 / window_size[0], y1 / window_size[1]),
                                              (x2 / window_size[0], y2 / window_size[1]))

            window.add_image_with_padding(new_image)

    cv2.setMouseCallback(window.window, mouse_callback)

    while True:
        key = window.wait_key(60)
        window_size = window.get_size()
        x1, y1, x2, y2 = current_frame_coords
        image = window.load_current_image()

        if key == 27:
            break

        elif key == ord('w'):
            pixel_size = min(pixel_size + 2, 50)
            if current_frame_coords[0] != -1:

                preview = pixelate_region(image,
                                                       (x1 / window_size[0], y1 / window_size[1]),
                                                       (x2 / window_size[0], y2 / window_size[1]),
                                                       pixel_size)

                window.add_image_with_padding(preview)
                clone = preview.copy()
        elif key == ord('s'):
            pixel_size = max(pixel_size - 2, 2)
            if current_frame_coords[0] != -1:

                preview = pixelate_region(image,
                                               (x1 / window_size[0], y1 / window_size[1]),
                                               (x2 / window_size[0], y2 / window_size[1]),
                                               pixel_size)

                window.add_image_with_padding(preview)
                clone = preview.copy()

        window.add_image_with_padding(clone)


    cv2.setMouseCallback(window.window, lambda *args: None)
    window.add_image_with_padding(window.load_current_image())