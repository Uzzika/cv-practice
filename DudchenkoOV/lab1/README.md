# Практическая работа №1. Обработка изображений с использованием OpenCV

## Цель работы
Реализовать набор фильтров для обработки изображений, используя Python, OpenCV и NumPy. Основные эффекты и преобразования выполняются вручную на уровне массивов, без использования готовых высокоуровневых фильтров OpenCV. Реализация и набор доступных эффектов соответствуют файлу `first_lab.py`.

## Состав проекта
- `image_processing.py` — основной скрипт с реализацией фильтров и запуском через аргументы командной строки.
- `README.md` — краткое описание реализованных эффектов и примеры запуска.

## Используемые функции
В файле `image_processing.py` реализованы следующие функции обработки изображения:

- `change_resolution_nn(image, target_width, target_height)` — изменение размера изображения методом ближайшего соседа.
- `apply_sepia_bgr(image_bgr)` — применение эффекта сепии.
- `apply_vignette(image_bgr, strength=0.8)` — применение эффекта виньетки.
- `apply_pixelation(image, area_x=0, area_y=0, area_width=10, area_height=10, pixel_size=10)` — пикселизация выбранной области изображения.
- `apply_frame(image, border_size=10, border_color=(255, 255, 255))` — добавление простой прямоугольной рамки.
- `apply_figure_frame(image, frame_path)` — наложение декоративной рамки из отдельного изображения.
- `apply_lens_flare(image, flare_path, intensity=0.7, position=None)` — добавление блика из изображения.
- `watercolor_texture(image, intensity=1.0)` — наложение текстуры акварельной бумаги.

## Поддерживаемые фильтры
Скрипт поддерживает следующие значения аргумента `--filter`:

- `resize`
- `sepia`
- `vignette`
- `pixelate`
- `rect_border`
- `decor_border`
- `lens_flare`
- `watercolor`
- `all`

## Запуск программы
Общий формат запуска:

```bash
python image_processing.py --image <путь_к_изображению> --filter <название_фильтра> [дополнительные параметры]
```

## Примеры запуска

### Изменение размера
```bash
python image_processing.py --image test_image.jpg --filter resize --new_w 400 --new_h 300
```

### Сепия
```bash
python image_processing.py --image test_image.jpg --filter sepia
```

### Виньетка
```bash
python image_processing.py --image test_image.jpg --filter vignette --strength 0.7
```

### Пикселизация
```bash
python image_processing.py --image test_image.jpg --filter pixelate --x 50 --y 50 --w 200 --h 150 --pixel_size 12
```

### Простая рамка
```bash
python image_processing.py --image test_image.jpg --filter rect_border --border_width 20 --border_b 255 --border_g 255 --border_r 255
```

### Декоративная рамка из изображения
```bash
python image_processing.py --image test_image.jpg --filter decor_border --border_texture frame.jpg
```

### Блик из изображения
```bash
python image_processing.py --image test_image.jpg --filter lens_flare --lens_flare_texture flare.jpg --intensity 0.8
```

### Блик с указанием позиции
```bash
python image_processing.py --image test_image.jpg --filter lens_flare --lens_flare_texture flare.jpg --intensity 0.8 --cx 100 --cy 50
```

### Акварельная текстура
```bash
python image_processing.py --image test_image.jpg --filter watercolor --texture_strength 0.7
```

## Краткое описание алгоритмов

### 1. Изменение размера — `change_resolution_nn`
Функция изменяет размер изображения методом ближайшего соседа. Для каждого пикселя нового изображения вычисляется соответствующая позиция в исходном изображении, после чего значение просто копируется.

### 2. Сепия — `apply_sepia_bgr`
Эффект сепии реализован через пересчёт цветовых каналов по стандартной формуле сепии. Обработка выполняется в формате BGR, который использует OpenCV.

### 3. Виньетка — `apply_vignette`
Создаётся маска затемнения по расстоянию пикселя от центра изображения. Чем дальше пиксель от центра, тем сильнее затемнение.

### 4. Пикселизация — `apply_pixelation`
Выбранная область сначала уменьшается, а затем увеличивается обратно методом ближайшего соседа. За счёт этого появляется эффект крупных пикселей.

### 5. Простая рамка — `apply_frame`
По краям изображения создаётся прямоугольная рамка заданной толщины и цвета.

### 6. Декоративная рамка — `apply_figure_frame`
Поверх исходного изображения накладывается отдельное изображение рамки. Если размеры не совпадают, рамка масштабируется под размер основной картинки.

### 7. Блик — `apply_lens_flare`
Поверх изображения накладывается отдельная картинка с бликом. Смешивание выполняется по маске яркости и коэффициенту интенсивности.

### 8. Акварельная текстура — `watercolor_texture`
На изображение накладывается текстура бумаги `paper.jpg`. Чем выше значение параметра `intensity`, тем заметнее акварельный эффект.

## Примечание
В примерах выше сохранены команды запуска именно в том виде, в котором они были предоставлены. При этом описание функций и параметров составлено по содержимому файла `first_lab.py`.
