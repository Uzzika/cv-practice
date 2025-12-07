import argparse
import sys
import cv2 as cv
import numpy as np


def cli_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode',
                        help='Mode (\'image\', \'video\', \'imgproc\', \'filters\')',
                        type=str,
                        dest='mode',
                        default='image')
    parser.add_argument('-i', '--image',
                        help='Path to an image',
                        type=str,
                        dest='image_path')
    parser.add_argument('-o', '--output',
                        help='Output image name',
                        type=str,
                        dest='output_image',
                        default='output.jpg')
    parser.add_argument('-v', '--video',
                        help='Path to a video file',
                        type=str,
                        dest='video_path')
    parser.add_argument('-f', '--filter',
                        help='Filter type (\'resize\', \'sepia\', \'vignette\', \'pixelate\', \'simple_frame\', \'fancy_frame\', \'lens_flare\', \'watercolor\')',
                        type=str,
                        dest='filter_type')
    parser.add_argument('--width',
                        help='Target width for resize',
                        type=int,
                        default=800)
    parser.add_argument('--height',
                        help='Target height for resize',
                        type=int,
                        default=600)
    parser.add_argument('--radius',
                        help='Radius for vignette',
                        type=float,
                        default=200.0)
    parser.add_argument('--frame_width',
                        help='Frame width for simple frame',
                        type=int,
                        default=20)
    parser.add_argument('--frame_color',
                        help='Frame color in BGR format (comma separated)',
                        type=str,
                        default='255,255,255')
    parser.add_argument('--frame_number',
                        help='Frame type (\'1\', \'2\', \'3\')',
                        type=str,
                        default='255,255,255')

    args = parser.parse_args()
    return args


# 1. Разрешение
def resize_image(image, new_width, new_height):
    height, width = image.shape[:2]

    x_coords = np.linspace(0, width - 1, new_width).astype(int)
    y_coords = np.linspace(0, height - 1, new_height).astype(int)

    x_grid, y_grid = np.meshgrid(x_coords, y_coords)
    resized = image[y_grid, x_grid]

    return resized


# 2. Сепия
def apply_sepia(image):
    sepia_matrix = np.array([
        [0.272, 0.534, 0.131],
        [0.349, 0.686, 0.168],
        [0.393, 0.769, 0.189]
    ])

    sepia = np.dot(image.astype(float), sepia_matrix.T)
    sepia = np.clip(sepia, 0, 255).astype(np.uint8)

    return sepia


# 3. Виньетка
def apply_vignette(image, radius):
    height, width = image.shape[:2]

    x = np.arange(width)
    y = np.arange(height)
    x_grid, y_grid = np.meshgrid(x, y)

    center_x, center_y = width // 2, height // 2
    dist = np.sqrt((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2)

    factor = 1.0 - (dist / radius)
    factor = np.clip(factor, 0.1, 1.0)

    vignette = (image.astype(float) * factor[:, :, np.newaxis]).astype(np.uint8)

    return vignette


# 4. Пикселизация
# Глобальные переменные для обработки мыши
rect_start = None
rect_end = None
drawing = False


def draw_rectangle(event, x, y, flags, param):
    global rect_start, rect_end, drawing
    processed_image = param
    if event == cv.EVENT_LBUTTONDOWN:
        if drawing == False:
            rect_start = (x, y)
            drawing = True
        else:
            drawing = False
            rect_end = (x, y)
            x1, y1 = rect_start
            x2, y2 = rect_end
            start_x = min(x1, x2)
            start_y = min(y1, y2)
            end_x = max(x1, x2)
            end_y = max(y1, y2)

            pixelated_result = pixelate_region(processed_image, start_x, start_y, end_x, end_y)
            processed_image[:, :] = pixelated_result[:, :]

            cv.imshow('original_image', processed_image)


def mode_pixelate(img):
    processed = img.copy()
    global rect_start, rect_end, drawing

    # Сбрасываем состояние
    rect_start = None
    rect_end = None
    drawing = False

    cv.namedWindow("original_image")
    cv.setMouseCallback("original_image", draw_rectangle, param=processed)
    cv.imshow('original_image', img)

    print("=== РЕЖИМ ПИКСЕЛИЗАЦИИ ===")
    print("1. Кликните левой кнопкой мыши для начала выделения")
    print("2. Кликните еще раз для завершения выделения")
    print("3. Пикселизация применится автоматически")
    print("4. Нажмите любую клавишу для завершения")

    cv.waitKey(0)
    cv.destroyAllWindows()

    return processed


def pixelate_region(image, start_x, start_y, end_x, end_y, pixel_size=20):
    pixelated = image.copy()

    start_x = max(0, start_x)
    start_y = max(0, start_y)
    end_x = min(image.shape[1], end_x)
    end_y = min(image.shape[0], end_y)

    region_height = end_y - start_y
    region_width = end_x - start_x

    small_height = max(1, region_height // pixel_size)
    small_width = max(1, region_width // pixel_size)

    small_region = cv.resize(
        pixelated[start_y:end_y, start_x:end_x],
        (small_width, small_height),
        interpolation=cv.INTER_LINEAR
    )

    pixelated_region = cv.resize(
        small_region,
        (region_width, region_height),
        interpolation=cv.INTER_NEAREST
    )

    pixelated[start_y:end_y, start_x:end_x] = pixelated_region

    return pixelated


# 5. Рамка
def add_simple_frame(image, frame_width, frame_color):
    framed = image.copy()
    height, width = image.shape[:2]

    mask = np.zeros((height, width), dtype=bool)

    mask[:frame_width, :] = True  # Верх
    mask[height - frame_width:, :] = True  # Низ
    mask[:, :frame_width] = True  # Лево
    mask[:, width - frame_width:] = True  # Право

    framed[mask] = frame_color

    return framed


# 6. Фигурная рамка
def add_fancy_frame(image, frame_number=1):
        frame = cv.imread(f"scr/frame{frame_number}.JPG")
        if frame is None:
            return image

        h, w = image.shape[:2]
        frame = cv.resize(frame, (w, h))

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        mask = gray < 190  # Все что темнее белого - рамка

        result = image.copy()
        result[mask] = frame[mask]

        return result


# 7. Блики
def add_flare(image, flare_path="scr/flare.JPG", intensity=0.6):
        flare = cv.imread(flare_path)
        if flare is None:
            raise FileNotFoundError(f"Текстура вспышки не найдена: {flare_path}")

        if flare.shape[:2] != image.shape[:2]:
            flare = cv.resize(flare, (image.shape[1], image.shape[0]))

        image_float = image.astype(np.float32) / 255.0
        flare_float = flare.astype(np.float32) / 255.0

        result = image_float + flare_float * intensity
        result = np.clip(result, 0, 1)

        return (np.clip(result, 0, 1) * 255).astype(np.uint8)


# 8. Акварельная бумага
def add_watercolor_texture(image, texture_path="scr/aquarell_paper.JPG", intensity=0.9, strength=0.9):

    texture = cv.imread(texture_path)
    if texture is None:
        raise FileNotFoundError(f"Текстура не найдена по пути: {texture_path}")

    if texture.shape[:2] != image.shape[:2]:
        texture = cv.resize(texture, (image.shape[1], image.shape[0]))

    texture_gray = cv.cvtColor(texture, cv.COLOR_BGR2GRAY)
    texture_mask = 1 - (texture_gray.astype(np.float32) / 255.0)
    texture_mask = np.power(texture_mask, 1.0 / strength)
    texture_mask = texture_mask[:, :, np.newaxis]
    image_float = image.astype(np.float32)
    texture_float = texture.astype(np.float32)
    blended = (image_float * (1 - texture_mask * intensity) +
               texture_float * (texture_mask * intensity))

    return np.clip(blended, 0, 255).astype(np.uint8)


def apply_filter(image, filter_type, args):
    if filter_type == 'resize':
        return resize_image(image, args.width, args.height)
    elif filter_type == 'sepia':
        return apply_sepia(image)
    elif filter_type == 'vignette':
        return apply_vignette(image, args.radius)
    elif filter_type == 'pixelate':
        return mode_pixelate(image)
    elif filter_type == 'simple_frame':
        frame_color = np.array([int(c) for c in args.frame_color.split(',')])
        return add_simple_frame(image, args.frame_width, frame_color)
    elif filter_type == 'fancy_frame':
        return add_fancy_frame(image, args.frame_number)
    elif filter_type == 'lens_flare':
        return add_flare(image)
    elif filter_type == 'watercolor':
        return add_watercolor_texture(image)
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")


def filter_samples(image_path, output_image, filter_type, args):
    if image_path is None:
        raise ValueError('Empty path to the image')

    image = cv.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot load image from {image_path}")

    filtered_image = apply_filter(image, filter_type, args)

    cv.imshow('Original Image', image)
    cv.imshow('Filtered Image', filtered_image)
    cv.waitKey(0)

    cv.imwrite(output_image, filtered_image)
    print(f"Filtered image saved as {output_image}")

    cv.destroyAllWindows()


def highgui_image_samples(image_path, output_image):
    if image_path is None:
        raise ValueError('Empty path to the image')

    image = cv.imread(image_path)
    height, width, nchannels = image.shape

    cv.line(image, (0, 0), (width - 1, height - 1), (0, 255, 0), 5)
    cv.rectangle(image, (379, 216), (584, 423), (255, 0, 0), 5)
    cv.circle(image, (279, 292), 100, (0, 0, 255), 5)
    cv.putText(image, 'APPLE', (384, 235), cv.FONT_HERSHEY_COMPLEX,
               0.5, (255, 0, 0), 1, cv.LINE_AA)

    roi_x = 238
    roi_y = 238
    roi_width = roi_height = 100
    image_roi = image[roi_x:roi_x + roi_width, roi_y:roi_y + roi_height]
    image_roi_gray = np.zeros((roi_height, roi_width), np.uint8)
    cv.cvtColor(image_roi, cv.COLOR_BGR2GRAY, image_roi_gray)
    image[roi_x:roi_x + roi_width, roi_y:roi_y + roi_height, 0] = image_roi_gray
    image[roi_x:roi_x + roi_width, roi_y:roi_y + roi_height, 1] = image_roi_gray
    image[roi_x:roi_x + roi_width, roi_y:roi_y + roi_height, 2] = image_roi_gray

    cv.imshow('Init image', image)
    cv.waitKey(0)

    cv.imwrite(output_image, image)
    cv.destroyAllWindows()


def highgui_video_samples(video_path):
    if video_path is not None:
        capt = cv.VideoCapture(video_path)
    else:
        capt = cv.VideoCapture(0)

    if not capt.isOpened():
        raise ValueError('Path of the video file is incorrect '
                         'or camera is unavailable')

    while True:
        ret_code, frame = capt.read()
        if not ret_code:
            print('The next frame is unavailable (stream end)')
            break
        cv.imshow('Video', frame)
        if cv.waitKey(1) == ord('q'):
            break

    capt.release()
    cv.destroyAllWindows()


def imgproc_samples(image_path):
    if image_path is None:
        raise ValueError('Empty path to the image')
    src_image = cv.imread(image_path)

    gray_dst_image = cv.cvtColor(src_image, cv.COLOR_BGR2GRAY)
    cv.imshow('Gray image', gray_dst_image)
    cv.waitKey(0)

    blurred_image = cv.blur(src_image, (5, 5))
    cv.imshow('Blurred image', blurred_image)
    cv.waitKey(0)

    t = 190
    max_value = 255
    _, thresh_image = cv.threshold(gray_dst_image, t, max_value, cv.THRESH_BINARY)
    cv.imshow('Thresholded image', thresh_image)
    cv.waitKey(0)

    dilatation_shape = cv.MORPH_RECT
    dilatation_size = 1
    kernel = cv.getStructuringElement(dilatation_shape,
                                      (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                      (dilatation_size, dilatation_size))
    dilate_dst_image = cv.dilate(thresh_image, kernel)
    cv.imshow('Dilatation output', dilate_dst_image)
    cv.waitKey(0)

    edges = cv.Canny(image=blurred_image, threshold1=10, threshold2=190)
    cv.imshow('Canny edge detection', edges)
    cv.waitKey(0)

    cv.destroyAllWindows()


def main():
    args = cli_argument_parser()

    if args.mode == 'image':
        highgui_image_samples(args.image_path, args.output_image)
    elif args.mode == 'video':
        highgui_video_samples(args.video_path)
    elif args.mode == 'imgproc':
        imgproc_samples(args.image_path)
    elif args.mode == 'filters':
        if args.filter_type is None:
            raise ValueError('Filter type must be specified for filter mode')
        filter_samples(args.image_path, args.output_image, args.filter_type, args)
    else:
        raise ValueError('Unsupported mode value')


if __name__ == '__main__':
    sys.exit(main() or 0)