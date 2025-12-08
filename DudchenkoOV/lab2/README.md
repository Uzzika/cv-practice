# Практическая работа №2. Детектирование объектов на изображениях с использованием библиотеки OpenCV

## Цель работы

Разработать приложение для детектирования транспортных средств на изображениях с использованием модуля `cv2.dnn` библиотеки OpenCV и предобученных нейросетевых моделей. Реализовать иерархию детекторов, визуализировать результаты и оценить качество по метрикам **TPR** и **FDR**.

---

## 1. Данные

Для экспериментов использована последовательность кадров видеозаписи:

* Кадры: `data/imgs_MOV03478/`
* Разметка: `data/mov03478.txt`

Формат разметки:

```
frame_id CLASS x1 y1 x2 y2
```

Пример:

```
0 CAR 339 82 446 169
```

---

## 2. Структура проекта

```
lab2/
  demo.py               # Демонстрационное приложение
  detectors.py          # Иерархия классов детекторов
  metrics.py            # Реализация IoU, TPR, FDR
  README.md             # Отчёт
  
  data/
    imgs_MOV03478/      # Кадры
    mov03478.txt        # Разметка

  models/
    yolov3.cfg
    yolov3.weights
    yolov4-tiny.cfg
    yolov4-tiny.weights
    deploy.prototxt
    mobilenet_iter_73000.caffemodel
    ssd_classes.txt
```

---

## 3. Иерархия детекторов

Все детекторы наследуются от абстрактного класса:

```python
class BaseDetector(ABC):
    def __init__(self, class_names, vehicle_classes):
        self.class_names = class_names
        self.vehicle_classes = set(vehicle_classes)
```

Реализовано три детектора:

* `YOLOv3Detector`
* `YOLOv4TinyDetector`
* `SSDCaffeDetector`

Выбор модели в `demo.py` осуществляется параметром:

```
--model {yolov3, yolov4-tiny, ssd}
```

---

# 4. Модели и методы обработки

## 4.1. YOLOv3 (COCO)

### Файлы модели:

* `models/yolov3.cfg`
* `models/yolov3.weights`

### Предобработка:

* размер входа: `416×416`
* нормализация `1/255`
* перестановка каналов BGR → RGB
* формирование blob:

```python
blob = cv2.dnn.blobFromImage(img, 1/255.0, (416,416), swapRB=True, crop=False)
```

### Постобработка:

* получение выходов YOLO-голов
* выбор класса с максимальным score
* фильтрация `confidence > threshold`
* фильтрация по транспортным классам
* вычисление `x1, y1, x2, y2`
* NMS: `cv2.dnn.NMSBoxes`

---

## 4.2. YOLOv4-tiny (COCO)

### Файлы модели:

* `models/yolov4-tiny.cfg`
* `models/yolov4-tiny.weights`

YOLOv4-tiny использует тот же формат обработки, что YOLOv3.

### Предобработка:

идентична YOLOv3

### Постобработка:

идентична YOLOv3

**Преимущества:**  высокая скорость.

---

## 4.3. SSD MobileNet (Caffe)

### Файлы модели:

* `deploy.prototxt`
* `mobilenet_iter_73000.caffemodel`
* `ssd_classes.txt`

### Предобработка:

```python
blob = cv2.dnn.blobFromImage(
    cv2.resize(img, (300,300)),
    scalefactor=0.007843,
    size=(300,300),
    mean=127.5
)
```

### Постобработка:

* анализ массива формы `[1, 1, N, 7]`
* фильтрация по confidence
* перевод нормированных координат в пиксельные
* фильтрация транспортных классов

---

# 5. Метрики качества

## 5.1. IoU

[
IoU = \frac{|B_{pred} \cap B_{gt}|}{|B_{pred} \cup B_{gt}|}
]

Порог IoU: **0.5**

---

## 5.2. TP / FP / FN

* **TP** — модель нашла объект правильно
* **FP** — лишний бокс
* **FN** — объект не найден

---

## 5.3. Итоговые показатели

[
TPR = \frac{TP}{TP + FN}
]

[
FDR = \frac{FP}{TP + FP}
]

---

# 6. Приложение demo

Примеры запуска:

```bash
python demo.py --model yolov3 --show
python demo.py --model yolov4-tiny --show
python demo.py --model ssd --show
```

Выводит:

* TP, FP, FN по кадрам
* суммарные TPR, FDR
* визуализацию детекций

---

# 7. Результаты

| Модель        | TPR   | FDR   |
| ------------- | ----- | ----- |
| YOLOv3        | 0.XXX | 0.XXX |
| YOLOv4-tiny   | 0.XXX | 0.XXX |
| SSD MobileNet | 0.XXX | 0.XXX |

(вставьте реальные значения)

---

# 8. Выводы

1. Реализовано приложение для детектирования транспорта.
2. Использованы три модели: YOLOv3, YOLOv4-tiny, SSD.
3. YOLOv3 — наилучшая точность.
4. YOLOv4-tiny — оптимальный баланс.
5. SSD — самая быстрая, но менее точная.

---

# 9. Команды запуска

```bash
python demo.py --model yolov3
python demo.py --model yolov4-tiny
python demo.py --model ssd
```
