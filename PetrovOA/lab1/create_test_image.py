"""
Скрипт для создания тестового изображения
"""
import cv2
import numpy as np

# Создаем цветное изображение 800x600
width, height = 800, 600
test_image = np.zeros((height, width, 3), dtype=np.uint8)

# Градиентный фон
for i in range(height):
    for j in range(width):
        test_image[i, j] = [
            int(255 * i / height),           # B
            int(255 * j / width),            # G
            int(255 * (1 - i / height))      # R
        ]

# Добавляем несколько цветных кругов
cv2.circle(test_image, (200, 200), 80, (255, 0, 0), -1)    # Синий
cv2.circle(test_image, (400, 300), 100, (0, 255, 0), -1)   # Зеленый
cv2.circle(test_image, (600, 400), 90, (0, 0, 255), -1)    # Красный
cv2.circle(test_image, (300, 450), 70, (255, 255, 0), -1)  # Голубой
cv2.circle(test_image, (650, 150), 60, (255, 0, 255), -1)  # Пурпурный

# Добавляем текст
cv2.putText(test_image, 'Test Image', (250, 100), 
            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

# Сохраняем
cv2.imwrite('test_image.jpg', test_image)
print('Тестовое изображение создано: test_image.jpg')
