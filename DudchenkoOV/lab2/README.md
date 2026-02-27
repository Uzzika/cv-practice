Практическая работа №2. Детектирование объектов на изображениях с использованием библиотеки OpenCV
Цель работы

Разработать приложение для детектирования транспортных средств на последовательности кадров видео с использованием обученных нейронных сетей из "зоопарка" моделей OpenCV (модуль cv2.dnn).

Реализовать:

иерархию классов детекторов,

поддержку нескольких моделей,

предобработку входных изображений,

постобработку выхода сети,

вычисление метрик качества TPR и FDR,

визуализацию результатов.

Структура проекта
lab2/
│
├── demo.py              # демонстрационное приложение
├── detectors.py         # иерархия детекторов
├── metrics.py           # IoU и вычисление TPR/FDR
│
├── models/
│   ├── yolov3.cfg
│   ├── yolov3.weights
│   ├── yolov4-tiny.cfg
│   ├── yolov4-tiny.weights
│   ├── deploy.prototxt
│   ├── mobilenet_iter_73000.caffemodel
│   └── ssd_classes.txt
│
└── data/
    ├── imgs_MOV03478/   # кадры видео
    └── mov03478.txt     # разметка
Используемые модели

В работе реализована поддержка трёх моделей:

YOLOv3 (Darknet, COCO)

YOLOv4-tiny (Darknet, COCO)

SSD MobileNet (Caffe)

Все модели подключаются через модуль cv2.dnn.

Иерархия классов
BaseDetector (абстрактный класс)
    ├── YOLOv3Detector
    ├── YOLOv4TinyDetector
    └── SSDCaffeDetector
BaseDetector

Содержит:

список классов модели

список интересующих транспортных классов

абстрактный метод detect()

Предобработка изображений
1. YOLOv3 / YOLOv4-tiny

Предобработка выполняется через cv2.dnn.blobFromImage:

масштабирование к размеру 416×416

нормализация пикселей (деление на 255)

перестановка каналов BGR → RGB

blob = cv2.dnn.blobFromImage(
    image,
    scalefactor=1/255.0,
    size=(416, 416),
    swapRB=True,
    crop=False
)
2. SSD MobileNet (Caffe)

Используется другой формат входа:

размер 300×300

масштабирование 1/127.5

вычитание среднего 127.5

blob = cv2.dnn.blobFromImage(
    cv2.resize(image, (300, 300)),
    scalefactor=0.007843,
    size=(300, 300),
    mean=127.5
)
Постобработка выхода сети
YOLO

Выход сети представляет собой массив предсказаний:

[x_center, y_center, width, height, objectness, class_scores...]

Алгоритм постобработки:

Выбор класса с максимальной вероятностью

Проверка confidence > threshold

Перевод координат в формат (x1, y1, x2, y2)

Применение NMS (cv2.dnn.NMSBoxes)

Фильтрация по транспортным классам

SSD

Выход имеет формат:

[image_id, class_id, confidence, x1, y1, x2, y2]

Алгоритм:

Проверка confidence > threshold

Перевод нормированных координат в пиксели

Фильтрация по транспортным классам

Сопоставление с разметкой

Разметка имеет формат:

frame_id class_name x1 y1 x2 y2
Алгоритм сопоставления:

Для каждого детектированного объекта:

сравнение класса

вычисление IoU

Выбор GT с максимальным IoU

Если IoU ≥ порога (например 0.3):

True Positive

Иначе:

False Positive

Неиспользованные GT → False Negative

Метрики качества
1. TPR (True Positive Rate)
TPR = TP / (TP + FN)

Показывает долю правильно обнаруженных объектов.

2. FDR (False Discovery Rate)
FDR = FP / (TP + FP)

Показывает долю ложных детекций среди всех обнаружений.

Визуализация

При запуске с флагом --show:

Каждый bbox окрашен в цвет класса.

Внутри bbox отображается:

car 0.873

Над bbox отображается:

GT: car

или

GT: none
Запуск программы
YOLOv3
python demo.py --model yolov3 --frames_dir ./data/imgs_MOV03478 --annotations ./data/mov03478.txt --show
YOLOv4-tiny
python demo.py --model yolov4-tiny --frames_dir ./data/imgs_MOV03478 --annotations ./data/mov03478.txt --show
SSD
python demo.py --model ssd --frames_dir ./data/imgs_MOV03478 --annotations ./data/mov03478.txt --show
Полученные результаты:
Model: yolov3
TP: 19320, FP: 6672, FN: 972
TPR = 0.9521
FDR = 0.2567

Model: yolov4-tiny
TP: 18060, FP: 3711, FN: 2232
TPR = 0.8900
FDR = 0.1705

Model: ssd
TP: 11119, FP: 300, FN: 9173
TPR = 0.5479
FDR = 0.0263