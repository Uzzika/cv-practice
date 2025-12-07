# Практическая работа №1. Обработка изображений с OpenCV

---

## Описание

Скрипт `filters.py` реализует **набор фильтров обработки изображений** с использованием **OpenCV** и **NumPy**.  
Все фильтры реализованы **вручную**, без применения высокоуровневых функций OpenCV (`resize`, `filter2D`, `cvtColor`, `applyColorMap` и т.п.).  
Поддерживается **запуск из командной строки** с гибкими параметрами, включая **комментарии после `#`**.  
Изображения обрабатываются в формате **BGR** (как в OpenCV), результат возвращается в `uint8`.

---

## Функции

### 1. `resize_img(image, scale=None, h=None, w=None)`
**Изменение размера изображения методом ближайшего соседа.**

- `image` — входное изображение (BGR, `uint8`)
- `scale` — коэффициент масштабирования (например, `0.5`)
- `h`, `w` — новые размеры (если `scale=None`)

**Алгоритм**:
```
oh, ow = image.shape[:2]
if scale: h, w = int(oh*scale), int(ow*scale)
sy = oh / h; sx = ow / w
ys = (np.arange(h) * sy).astype(int)
xs = (np.arange(w) * sx).astype(int)
gy, gx = np.meshgrid(ys, xs, indexing='ij')
result = image[gy, gx]
```
> **Векторизованный подход** — без циклов, высокая производительность.

---

### 2. `sepia(image)`
**Эффект сепии (винтажный оттенок).**

**Матрица преобразования**:
```
[[0.272, 0.534, 0.131],
 [0.349, 0.686, 0.168],
 [0.393, 0.769, 0.189]]
```
**Алгоритм**:
```
f = image.astype(np.float32).reshape(-1, 3)
result = np.clip(f @ k.T, 0, 255).reshape(image.shape).astype(np.uint8)
```
> Полная векторизация через матричное умножение.

---

### 3. `vignette(image, strength=0.5)`
**Эффект виньетки — затемнение краёв.**

- `strength` — сила затемнения (чем больше — тем сильнее)

**Формула маски**:
```
mask = exp(-((X-w/2)^2)/(2*(w*strength)^2) + ((Y-h/2)^2)/(2*(h*strength)^2))
```
**Алгоритм**:
```
Y, X = np.ogrid[:h, :w]
mask = np.exp(-((X-w/2)**2)/(2*(w*strength)**2) + ((Y-h/2)**2)/(2*(h*strength)**2))
mask /= mask.max()
result = (image.astype(float) * mask[..., None]).clip(0, 255).astype(np.uint8)
```

---

### 4. `pixelate_region(image, x, y, w, h, size=10)`
**Пикселизация заданной области.**

- `x, y` — левый верхний угол области
- `w, h` — ширина и высота области
- `size` — размер блока пикселизации

**Алгоритм**:
```
for i in range(y, y+h, size):
    for j in range(x, x+w, size):
        i2, j2 = min(i+size, y+h), min(j+size, x+w)
        block = image[i:i2, j:j2]
        avg = block.mean(axis=(0,1)).astype(np.uint8)
        image[i:i2, j:j2] = avg
```
> Проверка границ, защита от выхода за пределы.

---

### 5. `pixelate_full(image, size=10)`
**Пикселизация всего изображения.**

- Вызывает `pixelate_region(0, 0, w, h, size)`.

---

### 6. `pixelate_interactive(image, size=10)`
**Интерактивная пикселизация.**

**Управление**:
Выделить область - **ЛКМ** (зажать → отпустить)
Увеличить блок - `+` или `=` 
Уменьшить блок - `-`
Сбросить изменения - `r` 
Выйти - `q` или `Esc` 

> **Рамка выделения**  
> **Предпросмотр в реальном времени**  
> **Callback**: `mouse_handler`

---

### 7. `rect_border(image, thick=30, color=(0,255,0))`
**Прямоугольная рамка по краям.**

- `thick` — толщина в пикселях
- `color` — цвет в BGR (по умолчанию зелёный)

**Алгоритм**:
```
out[:thick, :] = out[-thick:, :] = out[:, :thick] = out[:, -thick:] = color
```

---

### 8. `shape_border(image, thick=25, color=(255,0,0), kind='ellipse')`
**Фигурная рамка: эллипс, круг, ромб.**

- `kind`: `'ellipse'`, `'circle'`, `'diamond'`
- `color` — цвет рамки (по умолчанию красный)

**Уравнения маски**:
- **Ellipse**: `((x-cx)/iw)^2 + ((y-cy)/ih)^2 >= 1`
- **Circle**: `(x-cx)^2 + (y-cy)^2 >= r^2`
- **Diamond**: `|x-cx| + |y-cy| >= r`

---

### 9. `lens_flare(image, texture_path, center=(0.5,0.5), intensity=1.0)`
**Блик линзы через текстуру.**

- `texture_path` — путь к PNG/JPG (поддержка альфа-канала)
- `center` — `(x, y)` в долях от 0.0 до 1.0
- `intensity` — сила наложения

**Алгоритм**:
```
flare = resize_img(flare, h=fh, w=fw)
if alpha: rgb = flare_rgb * alpha[...,None] * intensity
result += rgb
result = clip(result, 0, 255).astype(np.uint8)
```
> **Только текстурный блик** — реалистично и надёжно.

---

### 10. `watercolor(image, texture_path)`
**Эффект акварельной бумаги.**

- `texture_path` — путь к текстуре бумаги
- Текстура масштабируется под размер изображения
- Добавляется с коэффициентом `0.5`

**Алгоритм**:
```
tex = resize_img(tex, h=h, w=w)
tex = tex.astype(float)/255.0 - tex.mean()
img_f = image.astype(float)/255.0
result = img_f + 0.5 * tex
result = (np.clip(result, 0, 1) * 255).astype(np.uint8)
```

---

## Запуск из командной строки

```
python filters.py <изображение> <фильтр> [параметры...]
```

---

## Примеры использования

```
# 1. Изменение размера
python filters.py image.jpg resize 0.5
python filters.py image.jpg resize 800 600

# 2. Сепия
python filters.py image.jpg sepia

# 3. Виньетка
python filters.py image.jpg vignette 0.5

# 4. Пикселизация
python filters.py image.jpg pixelate                    # интерактивный режим

# 5. Рамки
python filters.py image.jpg rect_border 40 255 0 255
python filters.py image.jpg shape_border 30 0 255 255 circle

# 6. Блик линзы
python filters.py image.jpg lens_flare flare_texture.jpg 0.7 0.3 1.5

# 7. Акварель
python filters.py image.jpg watercolor paper_texture.jpg