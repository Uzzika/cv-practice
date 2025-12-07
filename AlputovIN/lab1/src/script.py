import os
import cv2
import numpy as np
import argparse
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from filters import *

def parse_args():
    parser = argparse.ArgumentParser(description='Image filters demo')
    parser.add_argument('image', help='Image filename (without path)')
    parser.add_argument('filter', help='Filter type', choices=[
        'resize', 'sepia', 'vignette', 'pixelate', 'rect_border', 'shape_border', 'lens_flare', 'watercolor'])
    parser.add_argument('--params', nargs='+', help='Filter parameters', default=None)
    return parser.parse_args()

def main():
    args = parse_args()
    image_path = os.path.join(os.path.dirname(__file__), '../images', args.image)
    img = cv2.imread(image_path)
    if img is None:
        print(f'Error loading image! Check the path and file format: {image_path}')
        print('File exists:', os.path.exists(image_path))
        print('Current directory:', os.getcwd())
        sys.exit(1)
    result = None
    if args.filter == 'resize':
        h_orig, w_orig = img.shape[:2]
        if args.params:
            if args.params[0] == 'scale' and len(args.params) == 2:
                scale_factor = float(args.params[1])
                result = resize_image(img, scale_factor=scale_factor)
            elif len(args.params) >= 2:
                w, h = int(args.params[0]), int(args.params[1])
                result = resize_image(img, width=w, height=h)
            elif len(args.params) == 1:
                w = int(args.params[0])
                result = resize_image(img, width=w)
            else:
                print('Please specify width and height or scale!')
                sys.exit(1)
        else:
            print('Please specify parameters: width and height or scale!')
            sys.exit(1)
        h_new, w_new = result.shape[:2]
        print(f'Original resolution: {w_orig}x{h_orig}')
        print(f'New resolution: {w_new}x{h_new}')
    elif args.filter == 'sepia':
        result = sepia(img)
    elif args.filter == 'vignette':
        strength = float(args.params[0]) if args.params else 0.5
        result = vignette(img, strength)
    elif args.filter == 'pixelate':
        print('Select an area with the mouse for pixelation. Left button — select, Enter — apply.')
        clone = img.copy()
        rect = [0, 0, 0, 0]
        drawing = [False]
        def draw_rectangle(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                rect[0], rect[1] = x, y
                drawing[0] = True
            elif event == cv2.EVENT_MOUSEMOVE and drawing[0]:
                img_disp = clone.copy()
                cv2.rectangle(img_disp, (rect[0], rect[1]), (x, y), (0,255,0), 2)
                cv2.imshow('Select area', img_disp)
            elif event == cv2.EVENT_LBUTTONUP:
                rect[2], rect[3] = x, y
                drawing[0] = False
                img_disp = clone.copy()
                cv2.rectangle(img_disp, (rect[0], rect[1]), (rect[2], rect[3]), (0,255,0), 2)
                cv2.imshow('Select area', img_disp)
        cv2.namedWindow('Select area')
        cv2.setMouseCallback('Select area', draw_rectangle)
        cv2.imshow('Select area', clone)
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # Enter
                break
        cv2.destroyWindow('Select area')
        x1, y1 = min(rect[0], rect[2]), min(rect[1], rect[3])
        x2, y2 = max(rect[0], rect[2]), max(rect[1], rect[3])
        block_size = 10
        result = pixelate_area(img, x1, y1, x2, y2, block_size)
    elif args.filter == 'rect_border':
        color = tuple(map(int, args.params[:3])) if args.params and len(args.params) >= 3 else (0,255,0)
        thickness = int(args.params[3]) if args.params and len(args.params) > 3 else 10
        result = add_rect_border(img, color, thickness)
    elif args.filter == 'shape_border':
        color = tuple(map(int, args.params[:3])) if args.params and len(args.params) >= 3 else (255,0,0)
        thickness = int(args.params[3]) if args.params and len(args.params) > 3 else 10
        shape = args.params[4] if args.params and len(args.params) > 4 else 'circle'
        result = add_shape_border(img, color, thickness, shape)
    elif args.filter == 'lens_flare':
        if args.params and len(args.params) >= 3:
            center_x = float(args.params[0])
            center_y = float(args.params[1])
            intensity = float(args.params[2])
            result = lens_flare(img, center_x, center_y, intensity)
        else:
            result = lens_flare(img)
    elif args.filter == 'watercolor':
        strength = float(args.params[0]) if args.params else 0.3
        result = watercolor_texture(img, strength)
    else:
        print('Unknown filter!')
        sys.exit(1)

    h1, w1 = img.shape[:2]
    h2, w2 = result.shape[:2]
    max_h = max(h1, h2)
    if h1 != max_h:
        img_disp = cv2.resize(img, (w1 * max_h // h1, max_h))
    else:
        img_disp = img.copy()
    if h2 != max_h:
        result_disp = cv2.resize(result, (w2 * max_h // h2, max_h))
    else:
        result_disp = result.copy()
    combined = np.hstack((img_disp, result_disp))
    window_name = 'Original | Result'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, combined)
    print('Press any key to exit...')
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print('An error occurred:', e)
        import traceback
        traceback.print_exc()
