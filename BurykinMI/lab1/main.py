import cv2
import argparse
from filters import *


def main():
    parser = argparse.ArgumentParser(
        description="Применение фильтров к изображению"
    )

    parser.add_argument("image", help="Путь к изображению для обработки")
    parser.add_argument("filter",
                        help="Тип фильтра (resize, sepia, vignette, pixelate, frame, figframe, flare, texture)")
    parser.add_argument("params", nargs="*", help="Дополнительные параметры фильтра (опционально)")

    args = parser.parse_args()
    img = cv2.imread(args.image)

    if img is None:
        print("Ошибка: не удалось загрузить изображение.")
        return

    ftype = args.filter.lower()
    params = args.params
    result = None

    # Обработка выбранного фильтра
    if ftype == "resize":
        scale = float(params[0]) if params else 0.5
        result = resize_image(img, scale)

    elif ftype == "sepia":
        result = apply_sepia(img)

    elif ftype == "vignette":
        strength = float(params[0]) if params else 0.5
        result = apply_vignette(img, strength)

    elif ftype == "pixelate":
        # Интерактивный выбор области мышью
        pixel_size = int(params[0]) if params else 10
        print("Выберите область для пикселизации:")
        print("- Зажмите левую кнопку мыши и выделите область")
        print("- Отпустите кнопку для применения эффекта")
        print("- Нажмите 'q' или ESC для выхода")
        selector = PixelateSelector(img, pixel_size)
        result = selector.select_region()

    elif ftype == "frame":
        color = (0, 0, 255)
        thickness = int(params[0]) if params else 20
        result = add_rectangular_frame(img, color, thickness)

    elif ftype == "figframe":
        color = (0, 255, 0)
        thickness = int(params[0]) if params else 20
        frame_type = params[1] if len(params) > 1 else "wave"
        result = add_figure_frame(img, color, thickness, frame_type)

    elif ftype == "flare":
        if not params:
            print("Ошибка: для flare нужен путь к текстуре блика")
            print("Использование: python main.py image.jpg flare flare_texture.jpg [intensity]")
            return
        texture_path = params[0]
        intensity = float(params[1]) if len(params) > 1 else 0.7
        result = add_lens_flare(img, texture_path, intensity)

    elif ftype == "texture":
        if not params:
            print("Ошибка: для texture нужен путь к текстуре бумаги")
            print("Использование: python main.py image.jpg texture paper_texture.jpg [intensity]")
            return
        texture_path = params[0]
        intensity = float(params[1]) if len(params) > 1 else 0.3
        result = add_paper_texture(img, texture_path, intensity)

    else:
        print("Ошибка: неизвестный тип фильтра.")
        return

    if result is not None:
        # Отображение изображений
        cv2.imshow("Original", img)
        cv2.imshow("Result", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()