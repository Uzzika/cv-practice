import cv2
import numpy as np


def load_image(path: str):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot open or read image: {path}")
    return img


def _tick_pos(i, ticks, size):
    if i == 0:
        return 0
    if i == ticks:
        return size - 1
    return int(round(i * (size - 1) / ticks))


def _make_panel(
        img,
        ticks=6,
        tick_len=6,
        size_color=(0, 128, 255),
        axis_col=(0, 0, 0),
        bg=(255, 255, 255),
) -> np.ndarray:
    h, w, _ = img.shape

    (x_tw, x_th), _ = cv2.getTextSize(str(w - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
    (y_tw, y_th), _ = cv2.getTextSize(str(h - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
    ax_w = y_tw + tick_len + 10
    ax_h = x_th + tick_len + 10

    ph, pw = h + ax_h, ax_w + w
    p = np.full((ph, pw, 3), bg, np.uint8)

    p[0:h, ax_w:pw] = img

    cv2.line(p, (ax_w, 0), (ax_w, h), axis_col, 1, cv2.LINE_AA)
    cv2.line(p, (ax_w, h), (pw - 1, h), axis_col, 1, cv2.LINE_AA)

    for i in range(ticks + 1):
        xr = _tick_pos(i, ticks, w)
        x = ax_w + xr
        cv2.line(p, (x, h), (x, h + tick_len), axis_col, 1, cv2.LINE_AA)

        label = str(xr)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        if i == 0:
            tx = ax_w + 2
        elif i == ticks:
            tx = pw - tw - 2
        else:
            tx = x - tw // 2
            tx = max(ax_w + 2, min(pw - tw - 2, tx))
        ty = h + tick_len + th + 3
        cv2.putText(p, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.38, axis_col, 1, cv2.LINE_AA)

    for i in range(ticks + 1):
        yr = _tick_pos(i, ticks, h)
        cv2.line(p, (ax_w - tick_len, yr), (ax_w, yr), axis_col, 1, cv2.LINE_AA)

        label = str(yr)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        tx = max(2, ax_w - tick_len - 4 - tw)
        if i == 0:
            ty = th + 2
        elif i == ticks:
            ty = h - 2
        else:
            ty = max(th + 2, min(h - 2, yr + th // 2))
        cv2.putText(p, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.38, axis_col, 1, cv2.LINE_AA)

    cv2.putText(p, f"{w}x{h}", (ax_w + 6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, size_color, 1, cv2.LINE_AA)
    return p


def _compose_and_show(
        left_panel,
        right_panel,
        margin=28,
        gap=36,
        bg=(255, 255, 255),
        title_col=(0, 160, 0),
) -> None:
    H = max(left_panel.shape[0], right_panel.shape[0]) + 2 * margin
    W = left_panel.shape[1] + gap + right_panel.shape[1] + 2 * margin
    canvas = np.full((H, W, 3), bg, np.uint8)

    y0 = margin
    xL = margin
    xR = margin + left_panel.shape[1] + gap

    canvas[y0:y0 + left_panel.shape[0], xL:xL + left_panel.shape[1]] = left_panel
    canvas[y0:y0 + right_panel.shape[0], xR:xR + right_panel.shape[1]] = right_panel

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "Original", (xL + 6, y0 - 8), font, 0.8, title_col, 2, cv2.LINE_AA)
    cv2.putText(canvas, "Processed", (xR + 6, y0 - 8), font, 0.8, title_col, 2, cv2.LINE_AA)

    cv2.imshow("Filter", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_images(img_left, img_right):
    left_panel = _make_panel(img_left)
    right_panel = _make_panel(img_right)
    _compose_and_show(left_panel, right_panel)
