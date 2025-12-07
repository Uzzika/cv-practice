import cv2
import numpy as np
import sys

# глобальные переменные для пикселизации
start_pt = end_pt = None
drawing = False
orig_img = work_img = None
block_sz = 10

# обработчик мыши
def mouse_handler(event, x, y, flags, param):
    global start_pt, end_pt, drawing, work_img
    if event == cv2.EVENT_LBUTTONDOWN:
        start_pt = end_pt = (x, y)
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        end_pt = (x, y)
        temp = work_img.copy()
        cv2.rectangle(temp, start_pt, end_pt, (0, 0, 255), 2)  # красная рамка
        cv2.imshow("Выделение области", temp)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_pt = (x, y)
        x1, y1 = start_pt
        x2, y2 = end_pt
        rx = min(x1, x2)
        ry = min(y1, y2)
        rw = abs(x2 - x1)
        rh = abs(y2 - y1)
        if rw > 0 and rh > 0:
            work_img = pixelate_region(work_img, rx, ry, rw, rh, block_sz)
            cv2.imshow("Выделение области", work_img)
        start_pt = end_pt = None

# пикселизация области
def pixelate_region(img, x, y, w, h, size=10):
    out = img.copy()
    x2 = min(x + w, img.shape[1])
    y2 = min(y + h, img.shape[0])
    for i in range(y, y2, size):
        for j in range(x, x2, size):
            i2 = min(i + size, y2)
            j2 = min(j + size, x2)
            block = out[i:i2, j:j2]
            if block.size:
                avg = block.mean(axis=(0,1)).astype(np.uint8)
                out[i:i2, j:j2] = avg
    return out

# пикселизация всего изображения
def pixelate_full(img, size=10):
    return pixelate_region(img, 0, 0, img.shape[1], img.shape[0], size)

# интерактивная пикселизация
def pixelate_interactive(img, size=10):
    global orig_img, work_img, block_sz
    orig_img = img.copy()
    work_img = img.copy()
    block_sz = size
    cv2.namedWindow("Выделение области")
    cv2.setMouseCallback("Выделение области", mouse_handler)
    cv2.imshow("Выделение области", img)
    print("ЛКМ — выделить | +/− — размер | r — сброс | q — выход")
    while True:
        k = cv2.waitKey(1) & 0xFF
        if k in (ord('q'), 27): break
        if k == ord('r'):
            work_img = orig_img.copy()
            cv2.imshow("Выделение области", work_img)
        if k in (ord('+'), ord('=')):
            block_sz = min(50, block_sz + 2)
            print(f"Размер: {block_sz}")
        if k == ord('-'):
            block_sz = max(2, block_sz - 2)
            print(f"Размер: {block_sz}")
    cv2.destroyWindow("Выделение области")
    return work_img

# изменение размера
def resize_img(img, scale=None, h=None, w=None):
    oh, ow = img.shape[:2]
    if scale: h, w = int(oh*scale), int(ow*scale)
    if not h and not w: return img.copy()
    if not h: h = int(oh * w / ow)
    if not w: w = int(ow * h / oh)
    sy = oh / h; sx = ow / w
    ys = (np.arange(h) * sy).astype(int)
    xs = (np.arange(w) * sx).astype(int)
    gy, gx = np.meshgrid(ys, xs, indexing='ij')
    return img[gy, gx]

# сепия
def sepia(img):
    k = np.array([[0.272, 0.534, 0.131],
                  [0.349, 0.686, 0.168],
                  [0.393, 0.769, 0.189]])
    f = img.astype(np.float32).reshape(-1, 3)
    return np.clip(f @ k.T, 0, 255).reshape(img.shape).astype(np.uint8)

# виньетка
def vignette(img, strength=0.5):
    h, w = img.shape[:2]
    Y, X = np.ogrid[:h, :w]
    mask = np.exp(-(((X-w/2)**2)/(2*(w*strength)**2) + ((Y-h/2)**2)/(2*(h*strength)**2)))
    mask /= mask.max()
    return np.clip(img.astype(float) * mask[..., None], 0, 255).astype(np.uint8)

# прямоугольная рамка
def rect_border(img, thick=30, color=(0,255,0)):
    out = img.copy()
    out[:thick, :] = color
    out[-thick:, :] = color
    out[:, :thick] = color
    out[:, -thick:] = color
    return out

# фигурная рамка
def shape_border(img, thick=25, color=(255,0,0), kind='ellipse'):
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    cy, cx = h//2, w//2
    ih, iw = h//2 - thick, w//2 - thick
    for y in range(h):
        for x in range(w):
            if kind == 'ellipse' and ((x-cx)/iw)**2 + ((y-cy)/ih)**2 >= 1: mask[y,x] = 1
            if kind == 'circle' and (x-cx)**2 + (y-cy)**2 >= min(ih,iw)**2: mask[y,x] = 1
            if kind == 'diamond' and abs(x-cx) + abs(y-cy) >= min(ih,iw): mask[y,x] = 1
    out = img.copy()
    for c in range(3): out[:,:,c] = np.where(mask, color[c], out[:,:,c])
    return out

# блик через текстуру
def lens_flare(img, texture_path, center=(0.5,0.5), intensity=1.0):
    try:
        flare = cv2.imread(texture_path, cv2.IMREAD_UNCHANGED)
        if flare is None: raise FileNotFoundError
        h, w = img.shape[:2]
        scale = min(h,w)/3 / max(flare.shape[:2])
        fw = int(flare.shape[1] * scale)
        fh = int(flare.shape[0] * scale)
        flare = resize_img(flare, h=fh, w=fw)
        cx = int(center[0] * w)
        cy = int(center[1] * h)
        x1 = cx - fw//2; y1 = cy - fh//2
        result = img.astype(np.float32)
        alpha = flare[...,3]/255.0 if flare.shape[2]==4 else 1.0
        rgb = flare[...,:3].astype(float) * alpha[...,None] * intensity
        for i in range(fh):
            for j in range(fw):
                yi = y1 + i; xj = x1 + j
                if 0 <= yi < h and 0 <= xj < w:
                    result[yi, xj] += rgb[i,j]
        return np.clip(result, 0, 255).astype(np.uint8)
    except:
        print("Ошибка загрузки текстуры блика")
        return img.copy()

# акварельная бумага
def watercolor(img, texture_path):
    try:
        tex = cv2.imread(texture_path)
        if tex is None: raise FileNotFoundError
        tex = resize_img(tex, h=img.shape[0], w=img.shape[1])
        tex = tex.astype(np.float32)/255.0
        tex -= tex.mean()
        img_f = img.astype(np.float32)/255.0
        result = np.clip(img_f + 0.5 * tex, 0, 1)
        return (result * 255).astype(np.uint8)
    except:
        print("Ошибка загрузки текстуры бумаги")
        return img.copy()

# main
def main():
    if len(sys.argv) < 3:
        print("Использование: python filters.py <image> <filter> [args...]")
        print("Фильтры: resize, sepia, vignette, pixelate, rect_border, shape_border, lens_flare, watercolor")
        return

    path = sys.argv[1]
    cmd = sys.argv[2].lower()
    args = sys.argv[3:]

    # Убираем комментарии после # и пустые строки
    clean_args = []
    for a in args:
        if '#' in a:
            a = a.split('#', 1)[0].strip()
        if a:
            clean_args.append(a)
    args = clean_args

    img = cv2.imread(path)
    if img is None:
        print("Ошибка: изображение не найдено")
        return

    try:
        if cmd == 'resize':
            if len(args) == 1:
                result = resize_img(img, scale=float(args[0]))
            elif len(args) == 2:
                result = resize_img(img, h=int(args[0]), w=int(args[1]))
            else:
                result = img

        elif cmd == 'sepia':
            result = sepia(img)

        elif cmd == 'vignette':
            strength = float(args[0]) if args else 0.5
            result = vignette(img, strength)

        elif cmd == 'pixelate':
            if not args:
                result = pixelate_interactive(img)
            else:
                try:
                    if len(args) == 1:
                        size = int(args[0])
                        result = pixelate_full(img, size)
                    elif len(args) == 4:
                        x, y, w, h = map(int, args)
                        result = pixelate_region(img, x, y, w, h)
                    elif len(args) == 5:
                        x, y, w, h, size = map(int, args)
                        result = pixelate_region(img, x, y, w, h, size)
                    else:
                        print("pixelate: ожидается 0, 1, 4 или 5 аргументов")
                        result = pixelate_interactive(img)
                except ValueError:
                    print("Неверные параметры (ожидались числа). Запуск интерактивного режима...")
                    result = pixelate_interactive(img)

        elif cmd == 'rect_border':
            t = int(args[0]) if args else 30
            c = tuple(map(int, args[1:4])) if len(args) >= 4 else (0, 255, 0)
            result = rect_border(img, t, c)

        elif cmd == 'shape_border':
            t = int(args[0]) if args else 25
            c = tuple(map(int, args[1:4])) if len(args) >= 4 else (255, 0, 0)
            k = args[4] if len(args) > 4 else 'ellipse'
            result = shape_border(img, t, c, k)

        elif cmd == 'lens_flare':
            if not args:
                print("Нужно: flare_texture.jpg [x y] [intensity]")
                return
            tex = args[0]
            cx = float(args[1]) if len(args) > 1 else 0.5
            cy = float(args[2]) if len(args) > 2 else 0.5
            intens = float(args[3]) if len(args) > 3 else 1.0
            result = lens_flare(img, tex, (cx, cy), intens)

        elif cmd == 'watercolor':
            if not args:
                print("Нужно: paper_texture.jpg")
                return
            result = watercolor(img, args[0])

        else:
            print("Неизвестный фильтр")
            return

        # Показ результата
        if cmd != 'pixelate' or args:
            cv2.imshow("Оригинал", img)
            cv2.imshow(f"Фильтр: {cmd}", result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    except Exception as e:
        print("Ошибка:", e)

if __name__ == "__main__":
    main()