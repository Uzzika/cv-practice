import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def resize_image(image_path, width=None, height=None, scale=None, output_path=None):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Не удалось загрузить изображение: {image_path}")
    
    original_height, original_width = image.shape[:2]
    print(f"Исходный размер: {original_width}x{original_height}")
    
    if scale is not None:
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
    elif width is not None and height is not None:
        new_width = width
        new_height = height
    elif width is not None:
        ratio = width / original_width
        new_width = width
        new_height = int(original_height * ratio)
    elif height is not None:
        ratio = height / original_height
        new_height = height
        new_width = int(original_width * ratio)
    else:
        raise ValueError("Необходимо указать scale, width, height или width и height")
    
    print(f"Новый размер: {new_width}x{new_height}")
    
    x_indices = np.linspace(0, original_width - 1, new_width).astype(int)
    y_indices = np.linspace(0, original_height - 1, new_height).astype(int)
    
    x_grid, y_grid = np.meshgrid(x_indices, y_indices)
    resized_image = image[y_grid, x_grid]
    
    if output_path:
        cv2.imwrite(output_path, resized_image)
        print(f"Результат сохранен в: {output_path}")
    
    return resized_image

def apply_sepia(image, intensity=1.0):
    sepia_matrix = np.array([
        [0.131, 0.534, 0.272],
        [0.168, 0.686, 0.349],
        [0.189, 0.769, 0.393],
    ])
    
    sepia_matrix = sepia_matrix * intensity
    image_float = image.astype(np.float32)

    pixels = image_float.reshape(-1, 3)
    sepia_pixels = np.dot(pixels, sepia_matrix.T)
    
    sepia_image = sepia_pixels.reshape(image.shape)
    sepia_image = np.clip(sepia_image, 0, 255).astype(np.uint8)
    
    if intensity < 1.0:
        result = sepia_image.astype(np.float32) * intensity + image_float * (1.0 - intensity)
        sepia_image = np.clip(result, 0, 255).astype(np.uint8)
    
    return sepia_image

def apply_vignette(image, intensity=0.8):
    height, width = image.shape[:2]
    
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    x_grid, y_grid = np.meshgrid(x, y)
    radius = np.sqrt(x_grid**2 + y_grid**2)
    
    mask = 1.0 - radius * intensity
    mask = np.clip(mask, 0, 1)
    
    vignette_image = image.astype(np.float32) * mask[:, :, np.newaxis]
    vignette_image = np.clip(vignette_image, 0, 255).astype(np.uint8)
    
    return vignette_image

def apply_pixelation(image, x=0, y=0, width=10, height=10, pixel_size=10):
    pixelated_image = image.copy()
    img_height, img_width = image.shape[:2]
    
    x = max(0, min(x, img_width - 1))
    y = max(0, min(y, img_height - 1))
    width = min(width, img_width - x)
    height = min(height, img_height - y)
    
    region = pixelated_image[y:y+height, x:x+width]
    
    small_width = max(1, width // pixel_size)
    small_height = max(1, height // pixel_size)

    temp_input = "temp_region_input.jpg"
    temp_output = "temp_region_output.jpg"
    cv2.imwrite(temp_input, region)
    
    small_region = resize_image(temp_input, width=small_width, height=small_height)
    cv2.imwrite(temp_input, small_region)
    pixelated_region = resize_image(temp_input, width=width, height=height)
    pixelated_image[y:y+height, x:x+width] = pixelated_region
    
    if os.path.exists(temp_input):
        os.remove(temp_input)
    if os.path.exists(temp_output):
        os.remove(temp_output)

    pixelated_image[y:y+height, x:x+width] = pixelated_region
    return pixelated_image

def apply_frame(image, frame_width=10, frame_color=(255, 255, 255)):
    height, width = image.shape[:2]
    
    framed_image = np.full((height, width, 3), frame_color, dtype=np.uint8)
    framed_image[frame_width:height-frame_width, frame_width:width-frame_width] = image[frame_width:height-frame_width, frame_width:width- frame_width]
    
    return framed_image

def apply_figure_frame(image, frame_path):
    frame = cv2.imread(frame_path)
    if frame is None:
        raise ValueError(f"Не удалось загрузить рамку: {frame_path}")
    
    if frame.shape[:2] != image.shape[:2]:
        # Просто используем cv2.resize вместо вашей функции
        frame = cv2.resize(frame, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    
    result = image.copy().astype(np.float32)
    
    brightness = np.mean(frame, axis=2) / 255.0    
    alpha_mask = (brightness <= 0.9).astype(np.float32)    
    result = (image.astype(np.float32) * (1 - alpha_mask[:, :, np.newaxis]) + 
              frame.astype(np.float32) * alpha_mask[:, :, np.newaxis])
    
    return np.clip(result, 0, 255).astype(np.uint8)

def apply_lens_flare(image, flare_path, intensity=0.7, position=None):
    flare = cv2.imread(flare_path)
    if flare is None:
        raise ValueError(f"Не удалось загрузить блик: {flare_path}")
    
    flare_brightness = np.mean(flare, axis=2) / 255.0
    flare_alpha = np.where(flare_brightness > 0.1, flare_brightness * intensity, 0)
    flare_height, flare_width = flare.shape[:2]
    img_height, img_width = image.shape[:2]
    
    if position is None:
        pos_x = max(0, img_width - flare_width - 50)
        pos_y = 50
    else:
        pos_x, pos_y = position
    
    pos_x = max(0, min(pos_x, img_width - flare_width))
    pos_y = max(0, min(pos_y, img_height - flare_height))
    
    result_image = image.copy().astype(np.float32)    
    flare_end_y = min(pos_y + flare_height, img_height)
    flare_end_x = min(pos_x + flare_width, img_width)
    flare_region_height = flare_end_y - pos_y
    flare_region_width = flare_end_x - pos_x
    
    if flare_region_height <= 0 or flare_region_width <= 0:
        return image
    
    flare_cropped = flare[:flare_region_height, :flare_region_width]
    flare_alpha_cropped = flare_alpha[:flare_region_height, :flare_region_width]
    image_region = result_image[pos_y:flare_end_y, pos_x:flare_end_x]    
    background = image_region
    foreground = flare_cropped.astype(np.float32)
    
    screen_val = 255 - (255 - background) * (255 - foreground) / 255
    
    alpha_expanded = flare_alpha_cropped[:, :, np.newaxis]
    blended_region = background * (1 - alpha_expanded) + screen_val * alpha_expanded    
    result_image[pos_y:flare_end_y, pos_x:flare_end_x] = blended_region
    
    return np.clip(result_image, 0, 255).astype(np.uint8)

def watercolor_texture(image, intensity=1.0):
    texture_path = "/Users/taagordeeva/Desktop/cv-practice/GordeevaTS/lab1/watercolor_paper.jpg"
    texture = cv2.imread(texture_path)
    if texture is None:
        raise ValueError(f"Не удалось загрузить текстуру: {texture_path}")
    
    if texture.shape[:2] != image.shape[:2]:
        temp_path = "/tmp/temp_texture.jpg"
        cv2.imwrite(temp_path, texture)
        texture = resize_image(temp_path, width=image.shape[1], height=image.shape[0])
    
    texture_gray = 0.299 * texture[:, :, 2] + 0.587 * texture[:, :, 1] + 0.114 * texture[:, :, 0]
    texture_mask = 1 - (texture_gray / 255.0)    
    texture_mask = texture_mask[:, :, np.newaxis] * intensity
    
    blended = (image.astype(np.float32) * (1 - texture_mask) + 
               texture.astype(np.float32) * texture_mask)
    
    return np.clip(blended, 0, 255).astype(np.uint8)

def show_comparison(original_path, processed_image):
    original = cv2.imread(original_path)
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    processed_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)

    h1, w1 = original.shape[:2]
    h2, w2 = processed_image.shape[:2]
    dpi = 100
    fig_w = (w1 + w2) / dpi
    fig_h = max(h1, h2) / dpi
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    gs = fig.add_gridspec(1, 2, width_ratios=[w1, w2])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    ax1.imshow(original_rgb, interpolation='nearest')
    ax1.set_title(f'Оригинал: {w1}x{h1}')
    ax1.axis('off')
    ax1.set_aspect('equal') 

    ax2.imshow(processed_rgb, interpolation='nearest')
    ax2.set_title(f'Обработанный: {w2}x{h2}')
    ax2.axis('off')
    ax2.set_aspect('equal')

    plt.subplots_adjust(wspace=0.05, hspace=0)
    plt.show()

def parse_arguments():
    parser = argparse.ArgumentParser(description='Обработка изображений')
    parser.add_argument('image_path', help='Путь к исходному изображению')
    
    parser.add_argument('--width', type=int, help='Новая ширина')
    parser.add_argument('--height', type=int, help='Новая высота')
    parser.add_argument('--scale', type=float, help='Коэффициент масштабирования')
    
    parser.add_argument('--sepia', action='store_true', help='Применить эффект сепии')
    parser.add_argument('--sepia_intensity', type=float, default=1.0, 
                       help='Интенсивность эффекта сепии (0.0 - 1.0)')
    
    parser.add_argument('--vignette', action='store_true', help='Применить эффект виньетки')
    parser.add_argument('--vignette_intensity', type=float, default=0.8,
                       help='Интенсивность виньетки (0.0 - 1.0)')
    
    parser.add_argument('--pixelate', action='store_true', help='Применить пикселизацию')
    parser.add_argument('--pixel_size', type=int, default=10,
                       help='Размер пикселя (чем больше, тем более пиксельно)')
    
    parser.add_argument('--frame', action='store_true', help='Добавить прямоугольную рамку')
    parser.add_argument('--frame_width', type=int, default=20,
                       help='Ширина рамки в пикселях')
    parser.add_argument('--frame_color', type=str, default='255,255,255',
                       help='Цвет рамки в формате R,G,B (по умолчанию белый)')
    
    parser.add_argument('--figure_frame', action='store_true', help='Добавить фигурную рамку из файла')
    parser.add_argument('--frame_path', type=str, help='Путь к файлу с рамкой (PNG с прозрачностью)')
    
    parser.add_argument('--lens_flare', action='store_true', help='Добавить эффект бликов объектива')
    parser.add_argument('--flare_path', type=str, help='Путь к файлу с бликом (PNG с прозрачностью)')
    parser.add_argument('--flare_intensity', type=float, default=0.7,
                       help='Интенсивность блика (0.0 - 1.0)')
    parser.add_argument('--flare_x', type=int, help='X координата блика')
    parser.add_argument('--flare_y', type=int, help='Y координата блика')
    
    parser.add_argument('--watercolor', action='store_true', help='Применить эффект акварельной бумаги')
    parser.add_argument('--watercolor_intensity', type=float, default=1.0,
                       help='Интенсивность акварельной текстуры (0.0 - 1.0)')
    
    # For all
    parser.add_argument('--output', help='Путь для сохранения результата')
    parser.add_argument('--show', action='store_true', help='Показать сравнение изображений')
    
    return parser.parse_args()

def validate_arguments(args):
    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Файл {args.image_path} не существует")
    
    if args.figure_frame and not args.frame_path:
        raise ValueError("Для фигурной рамки необходимо указать --frame_path")
    
    if args.figure_frame and args.frame_path and not os.path.exists(args.frame_path):
        raise FileNotFoundError(f"Файл с рамкой {args.frame_path} не существует")
    
    if args.lens_flare and not args.flare_path:
        raise ValueError("Для эффекта бликов необходимо указать --flare_path")
    
    if args.lens_flare and args.flare_path and not os.path.exists(args.flare_path):
        raise FileNotFoundError(f"Файл с бликом {args.flare_path} не существует")
    
    # Проверка диапазонов интенсивностей
    for intensity_name in ['sepia_intensity', 'vignette_intensity', 'flare_intensity', 'watercolor_intensity']:
        intensity = getattr(args, intensity_name)
        if not 0.0 <= intensity <= 1.0:
            raise ValueError(f"Интенсивность {intensity_name} должна быть в диапазоне 0.0 - 1.0")
    
    return True

def main():
    try:
        args = parse_arguments()        
        validate_arguments(args)
        
        image = cv2.imread(args.image_path)
        if image is None:
            raise ValueError(f"Не удалось загрузить изображение: {args.image_path}")        
        result_image = image.copy()
        
        if args.width is not None or args.height is not None or args.scale is not None:
            print("Применяем изменение размера...")
            temp_path = "temp_input.jpg"
            cv2.imwrite(temp_path, result_image)
            result_image = resize_image(
                image_path=temp_path,
                width=args.width,
                height=args.height,
                scale=args.scale,
                output_path=None
            )
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        if args.sepia:
            print("Применяем эффект сепии...")
            result_image = apply_sepia(result_image, args.sepia_intensity)
        
        if args.vignette:
            print("Применяем эффект виньетки...")
            result_image = apply_vignette(result_image, args.vignette_intensity)
        
        if args.pixelate:
            win_title = "Highlight the area for pixelate this and click to 'Escape'."
            roi = cv2.selectROI(win_title, image, showCrosshair=False, printNotice=False)
            cv2.destroyWindow(win_title)

            x1, y1, w1, h1 = map(int, roi)
            if w1 > 0 and h1 > 0:
                result_image = apply_pixelation(
                    result_image, 
                    x=x1, 
                    y=y1, 
                    width=w1, 
                    height=h1, 
                    pixel_size=args.pixel_size
                )
        
        if args.frame:
            print(f"Добавляем прямоугольную рамку шириной {args.frame_width} пикселей...")
            try:
                frame_color = tuple(int(x) for x in args.frame_color.split(','))
                if len(frame_color) != 3:
                    raise ValueError
                frame_color = (frame_color[2], frame_color[1], frame_color[0])  # BGR
            except:
                print("Неверный формат цвета, используем белый по умолчанию")
                frame_color = (255, 255, 255)
            
            result_image = apply_frame(
                result_image, 
                frame_width=args.frame_width,
                frame_color=frame_color
            )
        
        if args.figure_frame:
            print(f"Добавляем фигурную рамку из файла {args.frame_path}...")
            result_image = apply_figure_frame(result_image, args.frame_path)
        
        if args.lens_flare:
            print(f"Добавляем эффект бликов из файла {args.flare_path}...")
            position = None
            if args.flare_x is not None and args.flare_y is not None:
                position = (args.flare_x, args.flare_y)
            
            result_image = apply_lens_flare(
                result_image,
                flare_path=args.flare_path,
                intensity=args.flare_intensity,
                position=position
            )
        
        if args.watercolor:
            print(f"Применяем эффект акварельной бумаги (интенсивность: {args.watercolor_intensity})...")
            result_image = watercolor_texture(result_image, args.watercolor_intensity)
        
        if args.output and result_image is not None:
            cv2.imwrite(args.output, result_image)
            print(f"Финальный результат сохранен в: {args.output}")
        
        if args.show and result_image is not None:
            show_comparison(args.image_path, result_image)
            
    except Exception as e:
        print(f"Ошибка: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
