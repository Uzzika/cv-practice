import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import os
from dataclasses import dataclass

def resize_image(image, width=None, height=None, scale_factor=None):
    h, w = image.shape[:2]
    
    if width is not None and height is not None:
        new_width = width
        new_height = height
    elif width is not None and height is None:
        ratio = width / w
        new_width = width
        new_height = int(h * ratio)
    elif height is not None and width is None:
        ratio = height / h
        new_width = int(w * ratio)
        new_height = height
    else:
        new_width = w
        new_height = h
    if scale_factor is not None:
        new_width = max(int(new_width * scale_factor), 1)
        new_height = max(int(new_height * scale_factor), 1)
    

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
        frame_width = int(min(h, w)/25)

    frame_image[0:frame_width] = [B,G,R]
    frame_image[-frame_width:] = [B,G,R]
    frame_image[frame_width:-frame_width,0:frame_width] = [B,G,R]
    frame_image[frame_width:-frame_width,-frame_width:] = [B,G,R]

    return frame_image


def add_figure_frame(image, frame_filename="", threshold = 30.0):
    frame = cv2.imread("src/" + str(frame_filename))
    if frame is None:
        print(f"Error: cannot load {frame_filename}")
        return image
    
    h, w = image.shape[:2]

    if frame.shape != image.shape:
    	frame = resize_image(frame, width=w, height=h)

    background_color = np.array([255, 255, 255])

    color_diff = np.sqrt(np.sum((frame.astype(np.float32) - background_color.astype(np.float32)) ** 2, axis=2))
    frame_mask = (color_diff > threshold).astype(np.uint8)
    frame_mask = frame_mask[:,:,np.newaxis]
    
    frame = image * (1 - frame_mask) + frame * frame_mask
    
    return frame


def add_glare(image, glare_path, strength=0.5, scale=0.5, center=None):
    glared_image = image.astype(np.float32)
    h_img, w_img = image.shape[:2]
    
    if center is None:
        center_x, center_y = w_img // 2, h_img // 2
    else:
        center_x, center_y = center
    
    glare = resize_image(cv2.imread("src/" + glare_path), scale_factor=scale)
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

def interactive_pixelation(image, pixel_size=10):
    temp_image = image.copy()
    cur_image = image.copy()
    selected_region = None

    def mouse_callback(event, x, y, flags, param):
        nonlocal selected_region, temp_image, cur_image
        if event == cv2.EVENT_LBUTTONDOWN:
            selected_region = [x, y, x, y]
        elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
            if selected_region:
                selected_region[2] = x
                selected_region[3] = y
                temp_image = cur_image.copy()
                cv2.rectangle(temp_image, 
                            (selected_region[0], selected_region[1]),
                            (selected_region[2], selected_region[3]), 
                            (256, 128, 128), 2)
        elif event == cv2.EVENT_LBUTTONUP:
            if selected_region:
                x1, y1, x2, y2 = selected_region
                x = (x1 + x2) // 2
                y = (y1 + y2) // 2
                width = abs(x2 - x1)
                height = abs(y2 - y1)

                if width > 0 and height > 0:
                    result = pixelate_region(cur_image, x, y, width, height, pixel_size)
                    cur_image = result
                    temp_image = result
        return None

    cv2.imshow('Interactive Pixelation', temp_image)
    cv2.setMouseCallback('Interactive Pixelation', mouse_callback)

    print("Instructions:")
    print("1) Click and drag to select region")
    print("2) Press 'q' to quit without changes")
    print("3) Press 'a' to save pixelation result")

    while True:
        cv2.imshow('Interactive Pixelation', temp_image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            cv2.destroyAllWindows()
            return image
        elif key == ord('a') and selected_region:
            x1, y1, x2, y2 = selected_region
            x = (x1 + x2) // 2
            y = (y1 + y2) // 2
            width = abs(x2 - x1)
            height = abs(y2 - y1)

            if width > 0 and height > 0:
                result = pixelate_region(cur_image, x, y, width, height, pixel_size)
                cv2.destroyAllWindows()
                return result

    cv2.destroyAllWindows()
    return image

def apply_filter(image, filter_function, **kwargs):
    return filter_function(image, **kwargs)


def display_images(original, filtered, title1="Original/Filtered Image"):
    my_alpha = 1.0
    def update(val):
        my_alpha = slider_alpha.val
        ax1.clear()
        ax1.imshow(original_rgb)
        ax1.imshow(filtered_rgb, alpha=my_alpha)
        ax1.set_title(title1)
        ax1.axis('off')
        fig.canvas.draw_idle()
    
    cvt_matrix = np.array([
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0]
    ])
    original_rgb = original @ cvt_matrix
    filtered_rgb = filtered @ cvt_matrix
    
    fig, ax1 = plt.subplots(1, 1, figsize=(15, 5))
    
    ax1.imshow(original_rgb)
    ax1.imshow(filtered_rgb, alpha=my_alpha)
    ax1.set_title(title1)
    ax1.axis('off')
    ax1_slider = plt.axes([0.2, 0.1, 0.65, 0.03])
    slider_alpha = Slider(ax1_slider, 'Opacity', 0.0, 1.0, valinit=1.0)
    slider_alpha.on_changed(update)

    plt.tight_layout()
    plt.show()

@dataclass
class Args:
    image_path: str = None
    filter_type: str = None
    frame_filename: str = None
    glare_path: str = None
    new_width: int = None
    new_height: int = None
    center_x: int = None
    center_y: int = None
    frame_width: int = None
    scale: float = 1.0
    intensity: float = 1.0
    radius: float = 0.8
    frame_r: int = 0
    frame_g: int = 0
    frame_b: int = 0
    frame_threshold: float = 30.0
    strength: float = 0.9
    pixel_size: int = 10

def get(arg, ourtype):
    tcorrect = False
    while not tcorrect:
        tcorrect = True
        temp = input()
        if (temp != 'n' and temp != ''):
            try:
                arg = ourtype(temp)
            except Exception as e:
                tcorrect = false
    return arg
        

def parser():
    args = Args()
    print("Image path: ")
    args.image_path = input()
    correct = False
    while not correct:
        correct = True
        print("Filter type (resize, sepia, vignette, basic_frame, figure_frame, watercolor, glare, pixelate): ")
        args.filter_type = input()
        match args.filter_type:
            case "resize":
                print("New width: ")
                args.new_width = get(args.new_width, int)
                print("New height: ")
                args.new_height = get(args.new_height, int)
                print("Scale: ")
                args.scale = get(args.scale, float)
            case "sepia":
                print("Intensity: ")
                args.intensity = get(args.intensity, float)
            case "vignette":
                print("Intensity: ")
                args.intensity = get(args.intensity, float)
                print("Vignette radius: ")
                args.radius = get(args.radius, float)
                print("Center X: ")
                args.center_x = get(args.center_x, int)
                print("Center Y: ")
                args.center_y = get(args.center_y, int)
            case "basic_frame":
                print("Frame width: ")
                args.frame_width = get(args.frame_width, int)
                print("Frame Red: ")
                args.frame_r = get(args.frame_r, int)
                print("Frame Green: ")
                args.frame_g = get(args.frame_g, int)
                print("Frame Blue: ")
                args.frame_b = get(args.frame_b, int)
            case "figure_frame":
                print("Frame filename: ")
                args.frame_filename = get(args.frame_filename, str)
                print("Frame threshold: ")
                args.frame_threshold = get(args.frame_threshold, float)
            case "watercolor":
                print("Intensity: ")
                args.intensity = get(args.intensity, float)
                print("Strength: ")
                args.strength = get(args.strength, float)
            case "glare":
                print("Glare filename: ")
                args.glare_path = get(args.glare_path, str)
                print("Strength: ")
                args.strength = get(args.strength, float)
                print("Glare scale: ")
                args.scale = get(args.scale, float)
                print("Glare center X: ")
                args.center_x = get(args.center_x, int)
                print("Glare center Y: ")
                args.center_y = get(args.center_y, int)
            case "pixelate":
                print("Pixel size: ")
                args.pixel_size = get(args.pixel_size, int)
            case _:
                correct = False
                print("No such filter type")
    
    return args


def main():
    args = parser()

    if not os.path.isfile(args.image_path):
        print(f"Error: file '{args.image_path}' not found.")
        return
    
    try:
        original_image = cv2.imread(args.image_path)
        if original_image is None:
            print(f"Error: cannot load image '{args.image_path}'")
            return
    except Exception as e:
        print(f"Error while loading image: {e}")
        return
    
    print(f"Image loaded: {args.image_path}")
    print(f"Size: {original_image.shape[1]}x{original_image.shape[0]}")
    
    if args.filter_type == 'resize':
        filtered_image = apply_filter(
        original_image, 
        resize_image, 
        width=args.new_width, 
        height=args.new_height, 
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
    elif args.filter_type == 'basic_frame':
        filtered_image = apply_filter(
        original_image, 
        add_simple_frame,
        frame_width=args.frame_width,
        B=args.frame_b,
        G=args.frame_g,
        R=args.frame_r
        )
    elif args.filter_type == 'figure_frame':
        filtered_image = apply_filter(
        original_image, 
        add_figure_frame,
        frame_filename=args.frame_filename,
        threshold=args.frame_threshold
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
        glare_path=args.glare_path,
        strength=args.strength,
        scale=args.scale,
        center=center
        )
    elif args.filter_type == 'pixelate':
        filtered_image = apply_filter(
        original_image, 
        interactive_pixelation,
        pixel_size=args.pixel_size
        )

    if (args.filter_type != 'resize'):
        display_images(original_image, filtered_image)
    
    save_path = "filtered_" + os.path.basename(args.image_path)
    cv2.imwrite(save_path, filtered_image)
    print(f"Result saved as: {save_path}")
    
    
if __name__ == '__main__':
    while True:
        main()