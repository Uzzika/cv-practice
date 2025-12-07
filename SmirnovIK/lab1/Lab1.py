import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse

def resize_image(img, new_h=None, new_w=None, scale=None):
    h, w, c = img.shape
    
    if scale is not None:
        if scale == 1:
            return img.copy()
        new_h = int(h * scale)
        new_w = int(w * scale)
    
    scale_h = h / new_h
    scale_w = w / new_w

    y = (np.arange(new_h) * scale_h).astype(np.int32)
    x = (np.arange(new_w) * scale_w).astype(np.int32)
    x_neigh_index, y_neigh_index = np.meshgrid(x, y)

    res = img[y_neigh_index, x_neigh_index]
    return res
    

def sepia(img, k):
    B = img[:,:,0].astype(np.float32)
    G = img[:,:,1].astype(np.float32)
    R = img[:,:,2].astype(np.float32)

    R = np.clip(R + 2*k, 0, 255)
    G = np.clip(G + 0.5*k, 0, 255)
    B = np.clip(B - k, 0, 255)

    res = np.zeros_like(img, dtype=np.uint8)
    res[:,:,0] = B
    res[:,:,1] = G
    res[:,:,2] = R

    return res


def vinetka(img,k,radius,center):
    h,w,_ = img.shape
    cx,cy = center
    max_dist = np.sqrt((max(cx,w-cx))**2+(max(cy,h-cy))**2)
    res = np.zeros_like(img,np.uint8)
    coeffs = np.ones((h,w))

    x = (-np.arange(w)+cx)**2
    y = (-np.arange(h)+cy)**2
    X,Y = np.meshgrid(x,y)
    distanses = (X+Y)**0.5
    norm_distanses = distanses/max_dist
    mask = norm_distanses>radius

    coeffs[mask] = 1-k*(norm_distanses[mask]-radius)/(1-radius)
    coeffs = np.clip(coeffs, 0, 1)
    coeffs = coeffs[:,:,np.newaxis]
    res = (img*coeffs).astype(np.uint8)
    return res

def pixelize(img, oblast, k):
    (x0, y0), h, w = oblast
    res = np.copy(img)
    h = int((h // k) * k)
    w = int((w // k) * k)
    region = img[y0:y0+h, x0:x0+w]
    small = resize_image(region,scale = 1/k)
    region = resize_image(small,scale = k)
    res[y0:y0+h, x0:x0+w] = region.astype(np.uint8)
    return res

def rect_frame(img, thickness,color):
    new_h,new_w = img.shape[0]+2*thickness,img.shape[1]+2*thickness
    res = np.full((new_h, new_w, 3), color, dtype=np.uint8)
    res[thickness:thickness+img.shape[0],thickness:thickness+img.shape[1],:] = img
    return res
def rgb2gray(img):
    B = img[:,:,0].astype(np.float32)
    G = img[:,:,1].astype(np.float32)
    R = img[:,:,2].astype(np.float32)
    gray = 0.144*B + 0.587*G + 0.299*R
    return gray.astype(np.uint8)
def frame(img, frame, thickness=100):

    h, w, _ = img.shape

    corner_tl = frame[:thickness, :thickness]       # верх-лево
    corner_tr = frame[:thickness, -thickness:]      # верх-право
    corner_bl = frame[-thickness:, :thickness]      # низ-лево
    corner_br = frame[-thickness:, -thickness:]     # низ-право
    
    top = frame[:thickness, thickness:-thickness] 
    bottom = frame[-thickness:, thickness:-thickness]
    left = frame[thickness:-thickness, :thickness]
    right = frame[thickness:-thickness, -thickness:]


    top_resized = resize_image(top, thickness,w-2*thickness)
    bottom_resized = resize_image(bottom,thickness,w-2*thickness)
    left_resized = resize_image(left, h-2*thickness,thickness)
    right_resized = resize_image(right, h-2*thickness,thickness)

    result = np.copy(img)
    
    mask = rgb2gray(corner_tl)>20
    result[:thickness, :thickness][mask] = corner_tl[mask]
    
    mask = rgb2gray(corner_tr)>20
    result[:thickness, -thickness:][mask] = corner_tr[mask]
    
    mask = rgb2gray(corner_bl)>20
    result[-thickness:, :thickness][mask] = corner_bl[mask]
    
    mask = rgb2gray(corner_br)>20
    result[-thickness:, -thickness:][mask] = corner_br[mask]
    
    mask = rgb2gray(top_resized)>20
    result[:thickness, thickness:-thickness][mask] = top_resized[mask]
    
    mask = rgb2gray(bottom_resized)>20
    result[-thickness:, thickness:-thickness][mask] = bottom_resized[mask]
    
    mask = rgb2gray(left_resized)>20
    result[thickness:-thickness, :thickness][mask] = left_resized[mask]
    
    mask = rgb2gray(right_resized)>20
    result[thickness:-thickness, -thickness:][mask] = right_resized[mask]

    return result

def bliki(img, texture,k):
    h,w,_ = img.shape
    scaled_texture = resize_image(texture,h,w)
    res = np.clip((img + k*scaled_texture),0,255).astype(np.uint8)
    return res

def watercolor(img, texture, k):
    h, w, _ = img.shape  
    scaled_texture = resize_image(texture, h, w)
    tex = scaled_texture.astype(np.float32) / 255.0
    tex = tex - tex.mean()
    img_f = img.astype(np.float32) / 255.0
    res = np.clip(img_f + k * tex, 0, 1)
    return (res * 255).astype(np.uint8)

def cli_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image', help='Path to input image', type=str, required=True)
    parser.add_argument('-o', '--output', help='Path to output image', type=str, default='output.jpg')
    parser.add_argument('-f', '--func', help='Function to apply',
                        choices=['resize', 'sepia', 'vinetka', 'pixelize', 'rect_frame', 'frame', 'bliki', 'watercolor'],
                        required=True)
    parser.add_argument('--k', type=float, help='Parameter k')
    parser.add_argument('--radius', type=float, help='Radius for vinetka')
    parser.add_argument('--new_size', nargs=2, type=int, help='Size of new image for resize')
    parser.add_argument('--scale', type=float, help='Scale for resize')
    parser.add_argument('--thickness', type=int, help='Thickness for rect_frame/frame')
    parser.add_argument('--frame', type=str, help='Path to frame image')
    parser.add_argument('--texture', type=str, help='Path to texture image')
    parser.add_argument('--color', nargs=3, type=int, help='Color of the rectangle frame')
    return parser.parse_args()

def main():
    args = cli_argument_parser()
    img = cv2.imread(args.image)
    

    if args.func == 'resize':
        new_h, new_w = args.new_size or (1,1)
        result = resize_image(img, new_h, new_w, args.scale)

    elif args.func == 'sepia':
        result = sepia(img, args.k or 30)

    elif args.func == 'vinetka':
        points = []
        def select_area(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.clear()
                points.append((x, y))
        cv2.imshow("select", img)
        cv2.setMouseCallback("select", select_area, points)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        center = points[0]
        cx, cy = center
        result = vinetka(img, args.k or 0.5, args.radius or 0.5, (cx,cy))

    elif args.func == 'pixelize':
        points = []
        def select_area(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))
            elif event == cv2.EVENT_LBUTTONUP:
                points.append((x, y))

        cv2.imshow("select", img)
        cv2.setMouseCallback("select", select_area,points)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
               
        oblast = (points[0]),points[1][1]-points[0][1],points[1][0]-points[0][0]
        result = pixelize(img, oblast, args.k or 10)

    elif args.func == 'rect_frame':
        result = rect_frame(img, args.thickness or 50, args.color or (0,0,255))

    elif args.func == 'frame':
        frame_img = cv2.imread(args.frame)
        result = frame(img, frame_img, args.thickness or 100)

    elif args.func == 'bliki':
        texture = cv2.imread(args.texture) if args.texture else None
        result = bliki(img, texture, args.k or 0.9)

    elif args.func == 'watercolor':
        texture = cv2.imread(args.texture) if args.texture else None
        result = watercolor(img, texture, args.k or 0.5)

    else:
        raise ValueError("Unknown function")

    if args.output:
        cv2.imwrite(args.output, result)
    cv2.imshow("Original",img)
    cv2.imshow("Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

main()
