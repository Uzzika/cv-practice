import os

import cv2
import numpy as np


def create_filter(filter_type, image, **kwargs):
    if filter_type == 'resize':
        return _resize_image(
            image,
            width=kwargs.get('width'),
            height=kwargs.get('height'),
            scale=kwargs.get('scale'),
        )
    elif filter_type == 'sepia':
        return _sepia_filter(
            image,
            intensity=kwargs.get('intensity'),
        )
    elif filter_type == 'vignette':
        return _vignette_filter(
            image,
            strength=kwargs.get('strength'),
            radius=kwargs.get('radius')
        )
    elif filter_type == 'pixelate':
        return _pixelate_region(
            image,
            x=kwargs.get('x'),
            y=kwargs.get('y'),
            w=kwargs.get('w'),
            h=kwargs.get('h'),
            block=kwargs.get('block'),
        )
    elif filter_type == 'pixelate_interactive':
        return _roi_pixelate(
            image,
            block=kwargs.get('block'),
        )
    elif filter_type == 'simple_border':
        return _add_simple_border(
            image,
            border=kwargs.get('border'),
            color=tuple(kwargs.get('border_color')),
        )
    elif filter_type == 'border':
        return _add_border(
            image,
            border_id=kwargs.get('border_id'),
            borders_dir=kwargs.get('borders_dir'),
            opacity=kwargs.get('opacity'),
        )
    elif filter_type == 'flare':
        return _add_flare(
            image,
            flares_dir=kwargs.get('flares_dir'),
            opacity=kwargs.get('opacity'),
            scale=kwargs.get('scale'),
        )
    elif filter_type == 'paper':
        return _add_paper(
            image,
            paper_path=kwargs.get('paper_path'),
            strength=kwargs.get('strength'),
        )
    else:
        raise ValueError(f"Unsupported filter type: {filter_type}")


def _resize_image(img, width=None, height=None, scale=None):
    h0, w0, _ = img.shape

    if scale is not None:
        if scale <= 0:
            raise ValueError("scale must be > 0")
        new_w = int(w0 * scale)
        new_h = int(h0 * scale)

    else:
        if width is None and height is None:
            raise ValueError("Provide scale or at least one of width/height")

        if width is not None and height is None:
            new_w = int(width)
            new_h = h0

        elif height is not None and width is None:
            new_h = int(height)
            new_w = w0

        else:
            new_w, new_h = int(width), int(height)

        if new_w <= 0 or new_h <= 0:
            raise ValueError("Resulting size must be positive")

    x_idx = (np.arange(new_w) * w0 // new_w)
    y_idx = (np.arange(new_h) * h0 // new_h)

    out = np.take(img, y_idx, axis=0)
    out = np.take(out, x_idx, axis=1)
    return out


def _sepia_filter(image, intensity=1.0):
    sepia_matrix = np.array([
        [0.131, 0.534, 0.272],
        [0.168, 0.686, 0.349],
        [0.189, 0.769, 0.393],
    ])

    M = (1.0 - intensity) * np.eye(3) + intensity * sepia_matrix
    out_f = image @ M.T
    out = np.clip(out_f, 0, 255).astype(np.uint8)

    return out


def _vignette_filter(image, strength=0.8, radius=0.8):
    strength = float(np.clip(strength, 0.0, 1.0))
    radius = float(np.clip(radius, 0.0, 1.0))

    h, w, _ = image.shape
    cx, cy = w * 0.5, h * 0.5

    xs = np.linspace(0, w - 1, w, dtype=np.float32)
    ys = np.linspace(0, h - 1, h, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)

    nx = (X - cx) / (w * 0.5)
    ny = (Y - cy) / (h * 0.5)
    r = np.sqrt(nx * nx + ny * ny)

    mask = np.ones_like(r, dtype=np.float32)
    mask[r > radius] = 1.0 - strength * ((r[r > radius] - radius) / (1 - radius))
    mask = np.clip(mask, 1.0 - strength, 1.0)
    mask = mask[..., None]

    img_f = image.astype(np.float32)
    fill_f = np.array((0, 0, 0), dtype=np.float32).reshape(1, 1, 3)

    out_f = img_f * mask + fill_f * (1.0 - mask)

    out = np.clip(out_f, 0, 255).astype(np.uint8)
    return out


def _pixelate_region(image, x, y, w, h, block):
    if block <= 1:
        return image

    h0, w0, _ = image.shape

    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(w0, x + w)
    y1 = min(h0, y + h)
    if x1 <= x0 or y1 <= y0:
        return image

    out = image.copy()
    roi = out[y0:y1, x0:x1]

    small_w = max(1, roi.shape[1] // block)
    small_h = max(1, roi.shape[0] // block)

    small = _resize_image(roi, width=small_w, height=small_h)

    pixelated = _resize_image(small, width=roi.shape[1], height=roi.shape[0])

    out[y0:y1, x0:x1] = pixelated
    return out


def _roi_pixelate(image, block=16):
    win_title = "Выдели область (Enter — применить, Esc — отменить)"
    roi = cv2.selectROI(win_title, image, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(win_title)

    x, y, w, h = map(int, roi)
    if w <= 0 or h <= 0:
        return image.copy()

    return _pixelate_region(image, x, y, w, h, block=int(block))


def _add_simple_border(image, border, color=(0, 0, 0)):
    if border is None or border <= 0:
        return image

    h, w, _ = image.shape
    b = int(min(border, h // 2, w // 2))
    if b == 0:
        return image

    out = image.copy()
    col = np.array(color, dtype=out.dtype).reshape(1, 1, 3)

    out[:b, :, :] = col
    out[h - b:h, :, :] = col
    out[:, :b, :] = col
    out[:, w - b:w, :] = col

    return out


def _add_border(image, border_id, borders_dir, opacity=1.0):
    if border_id is None:
        raise ValueError("border_id must be provided")

    frame_path = os.path.join(borders_dir, f"border{int(border_id)}.png")
    frame_rgba = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
    if frame_rgba is None:
        raise FileNotFoundError(f"Cannot open frame file: {frame_path}")

    h, w, _ = image.shape
    fh, fw, _ = frame_rgba.shape
    if (fh, fw) != (h, w):
        frame_rgba = _resize_image(frame_rgba, width=w, height=h)

    frame_rgb = frame_rgba[..., :3].astype(np.float32)
    alpha = frame_rgba[..., 3].astype(np.float32) / 255.0
    alpha *= float(np.clip(opacity, 0.0, 1.0))
    alpha_3 = alpha[..., None]

    base = image.astype(np.float32)
    out = frame_rgb * alpha_3 + base * (1.0 - alpha_3)
    return np.clip(out, 0, 255).astype(np.uint8)


def _add_flare(image, flares_dir, opacity=1.0, scale=1.0):
    flare_path = os.path.join(flares_dir, "flare.jpg")
    flare = cv2.imread(flare_path, cv2.IMREAD_COLOR)
    if flare is None:
        raise FileNotFoundError(f"Cannot open flare file: {flare_path}")

    h, w, _ = image.shape
    fh, fw, _ = flare.shape

    scale = float(max(scale, 0.01))
    if abs(scale - 1.0) > 1e-3:
        new_w = max(1, int(fw * scale))
        new_h = max(1, int(fh * scale))
        flare = _resize_image(flare, width=new_w, height=new_h)
        fh, fw, _ = flare.shape

    x = (w - fw) // 2
    y = (h - fh) // 2
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w, x + fw), min(h, y + fh)
    if x1 <= x0 or y1 <= y0:
        return image.copy()

    sx0, sy0 = x0 - x, y0 - y
    sx1, sy1 = sx0 + (x1 - x0), sy0 + (y1 - y0)

    base = image[y0:y1, x0:x1].astype(np.float32) / 255.0
    flare_f = flare[sy0:sy1, sx0:sx1].astype(np.float32) / 255.0

    opacity = float(np.clip(opacity, 0.0, 1.0))

    out_roi = 1.0 - (1.0 - base) * (1.0 - flare_f * opacity)
    out_roi = np.clip(out_roi * 255.0, 0, 255).astype(np.uint8)

    out = image.copy()
    out[y0:y1, x0:x1] = out_roi
    return out


def _add_paper(image, paper_path, strength=0.6):
    if not os.path.exists(paper_path):
        raise FileNotFoundError(f"Cannot open paper texture: {paper_path}")

    paper = cv2.imread(paper_path, cv2.IMREAD_COLOR)
    if paper is None:
        raise FileNotFoundError(f"Cannot read paper texture: {paper_path}")

    H, W, _ = image.shape
    if paper.shape[:2] != (H, W):
        paper = _resize_image(paper, width=W, height=H)

    paper_gray = paper.astype(np.float32).mean(axis=2) / 255.0

    mask = 1.0 - paper_gray
    mask = np.power(mask, 1.0 / max(strength, 1e-6))
    mask = mask[..., None]

    img_f = image.astype(np.float32)
    tex_f = paper.astype(np.float32)

    blended = img_f * (1.0 - mask) + tex_f * mask
    return np.clip(blended, 0, 255).astype(np.uint8)
