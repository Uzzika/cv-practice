import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def resize_image(image, width=None, height=None, scale_factor=None):
    h, w = image.shape[:2]
    
    if scale_factor is not None:
        new_width = max(int(w * scale_factor), 1)
        new_height = max(int(h * scale_factor), 1)
    elif width is not None and height is not None:
        new_width = width
        new_height = height
    elif width is not None:
        ratio = width / w
        new_width = width
        new_height = int(h * ratio)
    elif height is not None:
        ratio = height / h
        new_width = int(w * ratio)
        new_height = height
    else:
        return image.copy()
    

    resized = np.zeros((new_height, new_width, 3), dtype=image.dtype)
    
    y_indices = np.arange(new_height)
    x_indices = np.arange(new_width)
    
    src_y = ((y_indices / new_height) * h).astype(int)
    src_x = ((x_indices / new_width) * w).astype(int)
    
    X, Y = np.meshgrid(src_x, src_y)
    
    resized[:, :, :] = image[Y, X, :]
    
    return resized



def sepia_filter(image, intensity=1.0):
    transpose_sepia_matrix = np.array([
        [0.272, 0.349, 0.393],
        [0.534, 0.686, 0.769],
        [0.131, 0.168, 0.189]
    ])
    
    transpose_sepia_matrix = transpose_sepia_matrix * intensity + np.eye(3) * (1 - intensity)
    
    sepia_image = image @ transpose_sepia_matrix
    
    sepia_image = np.clip(sepia_image, 0, 255)
    
    return sepia_image.astype(np.uint8)

def vignette_filter(image, intensity=0.8, radius=0.8, center=None):
    h, w = image.shape[:2]
    
    if center is None:
        center_x, center_y = w // 2, h // 2
    else:
        center_x, center_y = center
    
    y, x = np.ogrid[:h, :w]
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
     
    max_distance = max([distance[0][0], distance[0][w-1], distance[h-1][0], distance[h-1][w-1]])
    
    normalized_distance = distance / (max_distance * radius)
    
    vignette_mask = 1 - normalized_distance
    vignette_mask = np.clip(vignette_mask, 0, 1)
    vignette_mask = vignette_mask ** intensity
    
    vignette_mask = vignette_mask[:, :, np.newaxis]
    
    image = image * vignette_mask
    
    return image.astype(np.uint8)


def add_simple_frame(image, frame_width=None, B=0, G=0, R=0):
    frame_image = image.copy()
    h, w = image.shape[:2]

    if frame_width is None:
        frame_width = int(min(h, w)/10)

    frame_image[0:frame_width] = [B,G,R]
    frame_image[-frame_width:] = [B,G,R]
    frame_image[frame_width:-frame_width,0:frame_width] = [B,G,R]
    frame_image[frame_width:-frame_width,-frame_width:] = [B,G,R]

    return frame_image


def add_figure_frame(image, frame_number=0, threshold = 30.0):
    frame = cv2.imread("src/frame" + str(frame_number) + ".jpg")
    if frame is None:
        print(f"Ошибка: не удалось загрузить рамку {frame_path}")
        return image
    
    h, w = image.shape[:2]

    if frame.shape != image.shape:
    	frame = resize_image(frame, width=w, height=h)

    background_color_array = np.array([[255, 255, 255], [255, 255, 255], [255, 255, 255], [245, 245, 245], [255, 255, 255]])
    background_color = background_color_array[frame_number]

    color_diff = np.sqrt(np.sum((frame.astype(np.float32) - background_color.astype(np.float32)) ** 2, axis=2))
    frame_mask = (color_diff > threshold).astype(np.uint8)
    frame_mask = frame_mask[:,:,np.newaxis]
    
    frame = image * (1 - frame_mask) + frame * frame_mask
    
    return frame


def add_glare(image, strength=0.5, scale=0.5, center=None):
    glared_image = image.astype(np.float32)
    h_img, w_img = image.shape[:2]
    
    if center is None:
        center_x, center_y = 3 * (w_img // 4), h_img // 4
    else:
        center_x, center_y = center
    
    glare = resize_image(cv2.imread("src/glare.jpg"), scale_factor=scale)
    h_glare, w_glare = glare.shape[:2]
    
    y_start = center_y - h_glare // 2
    y_end = y_start + h_glare
    x_start = center_x - w_glare // 2
    x_end = x_start + w_glare
    
    img_y_start = max(0, y_start)
    img_y_end = min(h_img, y_end)
    img_x_start = max(0, x_start)
    img_x_end = min(w_img, x_end)
    
    glare_y_start = max(0, -y_start)
    glare_y_end = h_glare - max(0, y_end - h_img)
    glare_x_start = max(0, -x_start)
    glare_x_end = w_glare - max(0, x_end - w_img)
    
    glared_image[img_y_start:img_y_end, img_x_start:img_x_end] += glare[glare_y_start:glare_y_end, glare_x_start:glare_x_end] * strength

    return  np.clip(glared_image, 0, 255).astype(np.uint8)


def watercolor_texture(image, intensity=0.3, strength=0.9):
    texture = cv2.imread("src\\watercolor_paper.jpg")
    if texture.shape != image.shape:
        texture = resize_image(texture, image.shape[1], image.shape[0])
    texture_gray = np.mean(texture, axis=2)
    texture_mask = 1 - (texture_gray / 255.0)
    texture_mask = texture_mask ** (1 / strength)
    texture_mask = texture_mask[:, :, np.newaxis]
    blended = image.astype(np.float32) * (1 - texture_mask * intensity) + texture.astype(np.float32) * (texture_mask * intensity)
    return np.clip(blended, 0, 255).astype(np.uint8)





def pixelate_region(image, center_x, center_y, region_width, region_height, pixel_size=10):
    result = image.copy()
    
    x1 = max(0, center_x - region_width // 2)
    y1 = max(0, center_y - region_height // 2)
    x2 = min(image.shape[1], center_x + region_width // 2)
    y2 = min(image.shape[0], center_y + region_height // 2)
    
    region = image[y1:y2, x1:x2]
    
    if region.size == 0:
        return result
    
    small_region = resize_image(region, scale_factor=1.0/pixel_size)
    
    pixelated_region = resize_image(small_region, width=x2 - x1, height=y2 - y1)
    
    result[y1:y2, x1:x2] = pixelated_region
    
    return result

def interactive_pixelation(image, region_width=200, region_height=150, pixel_size=15):
    display_image = image.copy()
    
    mouse_x, mouse_y = image.shape[1] // 2, image.shape[0] // 2
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal mouse_x, mouse_y, display_image
        
        if event == cv2.EVENT_MOUSEMOVE:
            mouse_x, mouse_y = x, y
            display_image = pixelate_region(image, mouse_x, mouse_y, region_width, region_height, pixel_size)
    
    window_name = "Interactive Pixelation - Press ESC to exit, Q to save"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    print("Интерактивная пикселизация запущена!")
    print("Управление:")
    print("- Двигайте мышью для перемещения области пикселизации")
    print("- Нажмите ESC для выхода")
    print("- Нажмите S для сохранения результата")
    print("- Нажмите +/- для изменения размера пикселей")
    print("- Нажмите W/A/S/D для изменения размера области")
    
    output_image = display_image.copy()
    while True:
        cv2.imshow(window_name, display_image)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            break
        elif key == ord('q') or key == ord('Q'):
            output_image = display_image.copy()
        elif key == ord('+') or key == ord('='):
            pixel_size = min(50, pixel_size + 2)
            print(f"Размер пикселя: {pixel_size}")
        elif key == ord('-'):
            pixel_size = max(2, pixel_size - 2)
            print(f"Размер пикселя: {pixel_size}")
        elif key == ord('w') or key == ord('W'):
            region_height = min(image.shape[0], region_height + 10)
            print(f"Высота области: {region_height}")
        elif key == ord('s') or key == ord('S'):
            region_height = max(50, region_height - 10)
            print(f"Высота области: {region_height}")
        elif key == ord('a') or key == ord('A'):
            region_width = max(50, region_width - 10)
            print(f"Ширина области: {region_width}")
        elif key == ord('d') or key == ord('D'):
            region_width = min(image.shape[1], region_width + 10)
            print(f"Ширина области: {region_width}")
    
    cv2.destroyAllWindows()
    return output_image







def apply_filter(image, filter_function, **kwargs):
    return filter_function(image, **kwargs)

def display_images(original, filtered, title1="Original Image", title2="Filtered Image"):

    cvt_matrix = np.array([
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0]
    ])
    original_rgb = original @ cvt_matrix
    filtered_rgb = filtered @ cvt_matrix
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.imshow(original_rgb)
    ax1.set_title(title1)
    ax1.axis('off')
    
    ax2.imshow(filtered_rgb)
    ax2.set_title(title2)
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()

def parser():
    parser = argparse.ArgumentParser(description='Изменение разрешения изображения')
    parser.add_argument('image_path', type=str, help='Путь к изображению')

    subparsers = parser.add_subparsers(dest='filter_type', help='Тип фильтра', required=True)

    resize_parser = subparsers.add_parser('resize', help='Изменение размера')
    resize_parser.add_argument('--width', type=int, help='Новая ширина')
    resize_parser.add_argument('--height', type=int, help='Новая высота')
    resize_parser.add_argument('--scale', type=float, help='Коэффициент масштабирования')

    sepia = subparsers.add_parser('sepia', help='Применеие фотоэффекта сепии')
    sepia.add_argument('--intensity', type=float, default=1.0, help='Интенсивность применения сепии')

    vignette = subparsers.add_parser('vignette', help='Применеие фотоэффекта виньетки')
    vignette.add_argument('--intensity', type=float, default=0.7, help='Интенсивность виньетки')
    vignette.add_argument('--radius', type=float, default=0.8, help='Радиус виньетки')
    vignette.add_argument('--center_x', type=int, help='X координата центра виньетки')
    vignette.add_argument('--center_y', type=int, help='Y координата центра виньетки')

    frame = subparsers.add_parser('frame', help='Наложение прямоугольной одноцветной рамки')
    frame.add_argument('--width', type=int, default=None, help='Ширина рамки')
    frame.add_argument('--r', type=int, default=0, help='Красный цвет')
    frame.add_argument('--g', type=int, default=0, help='Зелёный цвет')
    frame.add_argument('--b', type=int, default=0, help='Синий цвет')

    figure_frame = subparsers.add_parser('figure_frame', help='Наложение рамки на изображение')
    figure_frame.add_argument('--number', type=int, default=0, help='Номер фигурной рамки')
    figure_frame.add_argument('--threshold', type=float, default=30.0, help='Критическое рассточние цветового различия')

    watercolor = subparsers.add_parser('watercolor', help='Наложение текстуры акварели')
    watercolor.add_argument('--intensity', type=float, default=1.0, help='Интенсивность текстуры')
    watercolor.add_argument('--strength', type=float, default=0.9, help='Сила наложения текстуры')

    glare = subparsers.add_parser('glare', help='Наложение блика')
    glare.add_argument('--strength', type=float, default=0.5, help='Сила наложения блика [0.0, 1.0]')
    glare.add_argument('--scale', type=float, default=0.5, help='Изменение размера блика')
    glare.add_argument('--center_x', type=int, help='X координата центра блика')
    glare.add_argument('--center_y', type=int, help='Y координата центра блика')

    pixelation_parser = subparsers.add_parser('pixelate', help='Интерактивная пикселизация')
    pixelation_parser.add_argument('--width', type=int, default=200, help='Ширина области пикселизации')
    pixelation_parser.add_argument('--height', type=int, default=150, help='Высота области пикселизации')
    pixelation_parser.add_argument('--size', type=int, default=10, help='Размер пикселя')

    return parser.parse_args()

def main():
    args = parser()

    if not os.path.isfile(args.image_path):
        print(f"Ошибка: файл '{args.image_path}' не найден.")
        return
    
    try:
        original_image = cv2.imread(args.image_path)
        if original_image is None:
            print(f"Ошибка: не удалось загрузить изображение '{args.image_path}'")
            return
    except Exception as e:
        print(f"Ошибка при загрузке изображения: {e}")
        return
    
    print(f"Загружено изображение: {args.image_path}")
    print(f"Исходный размер: {original_image.shape[1]}x{original_image.shape[0]}")
    
    if args.filter_type == 'resize':
        filtered_image = apply_filter(
        original_image, 
        resize_image, 
        width=args.width, 
        height=args.height, 
        scale_factor=args.scale
        )
    elif args.filter_type == 'sepia':
        filtered_image = apply_filter(
        original_image, 
        sepia_filter,
        intensity=args.intensity
        )
    elif args.filter_type == 'vignette':
        center=None
        if args.center_x is None or args.center_x < 0 or args.center_x >= original_image.shape[1]:
            args.center_x=None
        if args.center_y is None or args.center_y < 0 or args.center_y >= original_image.shape[0]:
            args.center_y=None
        if args.center_x is not None and args.center_y is not None:
            center = (args.center_x, args.center_y)
        filtered_image = apply_filter(
        original_image, 
        vignette_filter,
        intensity=args.intensity,
        radius=args.radius,
        center=center
        )
    elif args.filter_type == 'frame':
        filtered_image = apply_filter(
        original_image, 
        add_simple_frame,
        frame_width=args.width,
        B=args.b,
        G=args.g,
        R=args.r
        )
    elif args.filter_type == 'figure_frame':
        filtered_image = apply_filter(
        original_image, 
        add_figure_frame,
        frame_number=args.number,
        threshold=args.threshold
        )
    elif args.filter_type == 'watercolor':
        filtered_image = apply_filter(
        original_image,
        watercolor_texture,
        intensity=args.intensity,
        strength=args.strength
        )
    elif args.filter_type == 'glare':
        center=None
        if args.center_x is None or args.center_x < 0 or args.center_x >= original_image.shape[1]:
            args.center_x=None
        if args.center_y is None or args.center_y < 0 or args.center_y >= original_image.shape[0]:
            args.center_y=None
        if args.center_x is not None and args.center_y is not None:
            center = (args.center_x, args.center_y)
        filtered_image = apply_filter(
        original_image, 
        add_glare,
        strength=args.strength,
        scale=args.scale,
        center=center
        )
    elif args.filter_type == 'pixelate':
        filtered_image = apply_filter(
        original_image, 
        interactive_pixelation,
        region_width=args.width, 
        region_height=args.height, 
        pixel_size=args.size
        )

    display_images(original_image, filtered_image, "Original Image", "Filtered Image")
    
    save_path = "filtered_" + os.path.basename(args.image_path)
    cv2.imwrite(save_path, filtered_image)
    print(f"Результат сохранен как: {save_path}")

if __name__ == "__main__":
    main()
