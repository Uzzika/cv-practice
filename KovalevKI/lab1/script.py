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
        new_width, new_height = width, height
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
    sepia_matrix = np.array([
        [0.272, 0.534, 0.131],
        [0.349, 0.686, 0.168],
        [0.393, 0.769, 0.189]
    ])

    identity = np.eye(3)
    matrix = intensity * sepia_matrix + (1 - intensity) * identity

    sepia_img = image @ matrix.T
    return np.clip(sepia_img, 0, 255).astype(np.uint8)

def vignette_filter(image, intensity=0.8, radius=0.8, center=None):
    h, w = image.shape[:2]
    if center is None:
        center_x, center_y = w // 2, h // 2
    else:
        center_x, center_y = center

    y, x = np.ogrid[:h, :w]
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    max_dist = np.sqrt(max(center_x**2, (w - center_x)**2) + max(center_y**2, (h - center_y)**2))

    normalized = distance / (max_dist * radius)
    mask = 1 - normalized
    mask = np.clip(mask, 0, 1)
    mask = mask ** intensity
    mask = mask[:, :, np.newaxis]

    result = image.astype(np.float32) * mask
    return np.clip(result, 0, 255).astype(np.uint8)

def pixelate_region(image, center_x, center_y, region_width, region_height, pixel_size=10):
    result = image.copy()
    h, w = image.shape[:2]

    x1 = max(0, center_x - region_width // 2)
    y1 = max(0, center_y - region_height // 2)
    x2 = min(w, center_x + region_width // 2)
    y2 = min(h, center_y + region_height // 2)

    if x1 >= x2 or y1 >= y2:
        return result

    region = image[y1:y2, x1:x2]
    small = resize_image(region, scale_factor=1.0/pixel_size)
    pixelated = resize_image(small, width=x2-x1, height=y2-y1)

    result[y1:y2, x1:x2] = pixelated
    return result

def add_simple_frame(image, frame_width=None, B=0, G=0, R=0):
    result = image.copy()
    h, w = image.shape[:2]

    if frame_width is None:
        frame_width = min(h, w) // 10

    color = [B, G, R]

    result[:frame_width, :] = color
    result[-frame_width:, :] = color
    result[frame_width:-frame_width, :frame_width] = color
    result[frame_width:-frame_width, -frame_width:] = color

    return result

def add_figure_frame(image, frame_number=0, threshold=30.0):
    frame_path = f"source/frame{frame_number}.jpg"
    if not os.path.exists(frame_path):
        print(f"Файл рамки {frame_path} не найден. Пропуск.")
        return image

    frame = cv2.imread(frame_path)
    h, w = image.shape[:2]

    if frame.shape != image.shape:
        frame = resize_image(frame, width=w, height=h)

    background_colors = {
        0: [255, 255, 255],
        1: [255, 255, 255],
        2: [250, 250, 250],
        3: [255, 255, 255],
        4: [255, 255, 255]
    }
    bg_color = np.array(background_colors.get(frame_number, [255, 255, 255]))

    diff = np.linalg.norm(frame.astype(np.float32) - bg_color, axis=2)
    mask = (diff > threshold).astype(np.uint8)
    mask = mask[:, :, np.newaxis]

    result = image * (1 - mask) + frame * mask
    return result.astype(np.uint8)

def add_glare(image, strength=0.5, scale=0.5, center=None):
    glare_path = "source/glare.jpg"
    if not os.path.exists(glare_path):
        print("Файл блика 'sourсe/glare.jpg' не найден. Пропуск.")
        return image

    glare_img = cv2.imread(glare_path)
    h_img, w_img = image.shape[:2]

    if center is None:
        cx, cy = 3 * w_img // 4, h_img // 4
    else:
        cx, cy = center

    scaled_glare = resize_image(glare_img, scale_factor=scale)
    h_g, w_g = scaled_glare.shape[:2]

    y1 = cy - h_g // 2
    y2 = y1 + h_g
    x1 = cx - w_g // 2
    x2 = x1 + w_g

    img_y1 = max(0, y1)
    img_y2 = min(h_img, y2)
    img_x1 = max(0, x1)
    img_x2 = min(w_img, x2)

    gl_y1 = max(0, -y1)
    gl_y2 = h_g - max(0, y2 - h_img)
    gl_x1 = max(0, -x1)
    gl_x2 = w_g - max(0, x2 - w_img)

    result = image.astype(np.float32)
    result[img_y1:img_y2, img_x1:img_x2] += (
        scaled_glare[gl_y1:gl_y2, gl_x1:gl_x2].astype(np.float32) * strength
    )
    return np.clip(result, 0, 255).astype(np.uint8)

def watercolor_texture(image, intensity=1.0, strength=0.9):
    texture_path = "source/watercolor_paper.jpg"
    if not os.path.exists(texture_path):
        print("Текстура 'sourсe/watercolor_paper.jpg' не найдена. Пропуск.")
        return image

    texture = cv2.imread(texture_path)
    h, w = image.shape[:2]
    if texture.shape[:2] != (h, w):
        texture = resize_image(texture, width=w, height=h)

    tex_gray = np.mean(texture, axis=2) / 255.0
    mask = 1 - tex_gray
    mask = np.power(mask, 1 / strength)
    mask = mask[:, :, np.newaxis]

    blended = image.astype(np.float32) * (1 - mask * intensity) + \
              texture.astype(np.float32) * (mask * intensity)

    return np.clip(blended, 0, 255).astype(np.uint8)

def interactive_pixelation_gui(image):
    import cv2 as cv

    h, w = image.shape[:2]
    window_name = "Interactive Pixelation [ESC: exit, LMB: apply]"

    pixel_size = 10
    region_width = 150
    region_height = 150
    center_x, center_y = w // 2, h // 2

    def update_display():
        display_img = image.copy()
        
        x1 = max(0, center_x - region_width // 2)
        y1 = max(0, center_y - region_height // 2)
        x2 = min(w, center_x + region_width // 2)
        y2 = min(h, center_y + region_height // 2)

        if x1 < x2 and y1 < y2:
            region = image[y1:y2, x1:x2]
            small = resize_image(region, scale_factor=1.0 / pixel_size)
            pixelated = resize_image(small, width=x2 - x1, height=y2 - y1)
            display_img[y1:y2, x1:x2] = pixelated

        cv.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv.putText(display_img, f'Pixel size: {pixel_size}', (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv.putText(display_img, f'Size: {region_width}x{region_height}', (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv.imshow(window_name, display_img[:, :, ::-1])

    click_event = [False]

    def mouse_callback(event, x, y, flags, param):
        nonlocal center_x, center_y
        if event == cv.EVENT_MOUSEMOVE:
            center_x, center_y = x, y
            update_display()
        elif event == cv.EVENT_LBUTTONDOWN:
            click_event[0] = True

    cv.namedWindow(window_name)
    cv.setMouseCallback(window_name, mouse_callback)
    update_display()

    print("Управление:")
    print("- Двигайте мышью — перемещение области")
    print("- Клавиши 'A'/'D' — уменьшить/увеличить ширину области")
    print("- Клавиши 'W'/'S' — уменьшить/увеличить высоту области")
    print("- Клавиши '+' и '-' — изменение размера пикселя")
    print("- ЛКМ — применить фильтр")
    print("- ESC — выйти без применения")

    result_image = None

    while True:
        key = cv.waitKey(30) & 0xFF

        if key == 27:
            break
        elif click_event[0]:
            result_image = pixelate_region(
                image,
                center_x=center_x,
                center_y=center_y,
                region_width=region_width,
                region_height=region_height,
                pixel_size=pixel_size
            )
            break

        elif key == ord('a'):
            region_width = max(10, region_width - 10)
            update_display()
        elif key == ord('d'):
            region_width = min(w, region_width + 10)
            update_display()
        elif key == ord('w'):
            region_height = max(10, region_height - 10)
            update_display()
        elif key == ord('s'):
            region_height = min(h, region_height + 10)
            update_display()

        elif key == ord('+') or key == ord('='):
            pixel_size = min(50, pixel_size + 1)
            update_display()
        elif key == ord('-'):
            pixel_size = max(1, pixel_size - 1)
            update_display()

    cv.destroyAllWindows()

    return result_image, region_width, region_height, pixel_size

def display_images(original, filtered, title1="Original", title2="Filtered"):
    def bgr_to_rgb(img):
        return img[..., ::-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    ax1.imshow(bgr_to_rgb(original))
    ax1.set_title(title1, fontsize=14)
    ax1.axis('off')

    ax2.imshow(bgr_to_rgb(filtered))
    ax2.set_title(title2, fontsize=14)
    ax2.axis('off')

    plt.tight_layout()
    plt.show()


def get_args():
    parser = argparse.ArgumentParser(description="Практическая работа №1: Обработка изображений")

    parser.add_argument("image_name", type=str, help="Имя файла изображения (в папке images/)")
    parser.add_argument("filter_type", choices=[
        "resize", "sepia", "vignette", "pixelate",
        "frame", "figure_frame", "glare", "watercolor"
    ], help="Тип фильтра")

    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--scale", type=float)

    parser.add_argument("--intensity", type=float, default=1.0)
    parser.add_argument("--radius", type=float, default=0.8)
    parser.add_argument("--center_x", type=int)
    parser.add_argument("--center_y", type=int)

    parser.add_argument("--r", type=int, default=0)
    parser.add_argument("--g", type=int, default=0)
    parser.add_argument("--b", type=int, default=0)
    parser.add_argument("--frame_width", type=int)

    parser.add_argument("--number", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=30.0)

    parser.add_argument("--strength", type=float, default=0.5)

    parser.add_argument("--region_width", type=int, default=200)
    parser.add_argument("--region_height", type=int, default=150)
    parser.add_argument("--pixel_size", type=int, default=10)

    return parser.parse_args()

def main():
    args = get_args()

    image_path = os.path.join("images", args.image_name)

    if not os.path.isfile(image_path):
        print(f"Ошибка: файл '{image_path}' не найден.")
        return

    image = cv2.imread(image_path)
    if image is None:
        print(f"Ошибка: не удалось загрузить изображение '{image_path}'.")
        return

    print(f"Изображение загружено: {image_path}, размер: {image.shape[1]}x{image.shape[0]}")

    if args.filter_type == "resize":
        filtered = resize_image(image, args.width, args.height, args.scale)
        if args.scale is not None:
            filter_name = f"Resize (scale={args.scale})"
        elif args.width and args.height:
            filter_name = f"Resize ({args.width}x{args.height})"
        elif args.width:
            filter_name = f"Resize (width={args.width})"
        elif args.height:
            filter_name = f"Resize (height={args.height})"
        else:
            filter_name = "Resize"

    elif args.filter_type == "sepia":
        filtered = sepia_filter(image, args.intensity)
        filter_name = f"Sepia (intensity={args.intensity:.1f})"

    elif args.filter_type == "vignette":
        center = (args.center_x, args.center_y) if args.center_x and args.center_y else None
        filtered = vignette_filter(image, args.intensity, args.radius, center)
        cx = args.center_x if args.center_x else "center"
        cy = args.center_y if args.center_y else "center"
        filter_name = f"Vignette (intensity={args.intensity:.1f}, radius={args.radius:.1f}, center=({cx},{cy}))"

    elif args.filter_type == "pixelate":
        print("Запуск интерактивной пикселизации...")
        result = interactive_pixelation_gui(image)
        if result[0] is None:
            print("Пикселизация отменена.")
            filtered = image
            filter_name = "Pixelate (отменено)"
        else:
            filtered, region_width, region_height, pixel_size = result
            filter_name = f"Interactive Pixelate ({region_width}×{region_height}, pixel={pixel_size})"
    elif args.filter_type == "frame":
        width = args.frame_width or args.width or min(image.shape[:2])//10
        color = [args.b, args.g, args.r]
        filtered = add_simple_frame(image, width, args.b, args.g, args.r)
        filter_name = f"Frame (width={width}, RGB=[{args.r}, {args.g}, {args.b}])"

    elif args.filter_type == "figure_frame":
        filtered = add_figure_frame(image, args.number, args.threshold)
        filter_name = f"Figure Frame №{args.number} (threshold={args.threshold})"

    elif args.filter_type == "glare":
        center = (args.center_x, args.center_y) if args.center_x and args.center_y else None
        filtered = add_glare(image, args.strength, args.scale, center)
        cx = args.center_x if args.center_x else "center"
        cy = args.center_y if args.center_y else "center"
        filter_name = f"Glare (strength={args.strength}, scale={args.scale:}, center=({cx},{cy}))"

    elif args.filter_type == "watercolor":
        filtered = watercolor_texture(image, args.intensity, args.strength)
        filter_name = f"Watercolor (intensity={args.intensity:.1f}, strength={args.strength:.1f})"

    display_images(image, filtered, "Original Image", filter_name)
    
    output_filename = "filtered_" + os.path.basename(args.image_name)
    output_path = os.path.join("images", output_filename)  # Сохраняем в ту же папку
    cv2.imwrite(output_path, filtered)
    print(f"Результат сохранён: {output_path}")


if __name__ == "__main__":
    main()