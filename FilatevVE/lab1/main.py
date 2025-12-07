import cv2
import numpy as np
from lib.manager_window import OpenCVWindowManager
from lib.change_images import *
from lib.manager_change import *

def print_manual():
    print("\n\n\nУправление:")
    print("Стрелка 'b'  - предыдущее изображение")
    print("Стрелка 'n' - следующее изображение")
    print("Нажмите 'i' - показать информацию о текущем изображении")
    print("Нажмите 'Esc' - выход")

    print("\n Фильтры:")
    print("1 Фотоэффект сепия")
    print("2 Фотоэффект виньетка")
    print("3 Пикселизация заданной прямоугольной области")
    print("4 Наложения прямоугольной одноцветной рамки")
    print("5 Наложения фигурной одноцветной рамки")
    print("6 Наложения эффекта бликов")
    print("7 Наложения текстуры акварельной бумаги")

if __name__ == "__main__":
    window = OpenCVWindowManager("OpenCV Image Viewer")
    window.set_size(800, 600)

    window.load_images_from_folder("images")

    print_manual()

    while True:
        key = window.wait_key(60)

        if key == ord('d'):
            window.show_next_image()

        elif key == ord('a'):
            window.show_previous_image()

        elif key == ord('i'):
            img_height, img_width = window.original_image.shape[:2]
            print(f"Информация об изображении:")
            print(f"  Размер: {img_width}x{img_height}")
            print(f"  Файл: {window.image_files[window.current_image_index]}")

        elif key == ord('1'):
            change_sepia(window)
            print_manual()

        elif key == ord('2'):
            change_apply_vignette_elliptical(window)
            print_manual()

        elif key == ord('3'):
            pixelate_region_manager(window)
            print_manual()

        elif key == ord('4'):
            changer_add_rectangular_border(window)
            print_manual()

        elif key == ord('5'):
            figure_frame_manager(window)
            print_manual()

        elif key == ord('6'):
            change_add_lens_flare(window)
            print_manual()

        elif key == ord('7'):
            change_add_watercolor_texture(window)
            print_manual()

        elif key == 27:
            break

    cv2.destroyAllWindows()