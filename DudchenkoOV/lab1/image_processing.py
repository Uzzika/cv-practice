import cv2
import numpy as np
import math

def change_resolution(image, new_width, new_height):
    old_height, old_width = image.shape[:2]

    scale_x = old_width / new_width
    scale_y = old_height / new_height

    if len(image.shape) == 3:
        resized_image = np.zeros((new_height, new_width, image.shape[2]), dtype = image.dtype)
    else:
        resized_image = np.zeros((new_height, new_width), dtype = image.dtype)

    for y in range(new_height):
        for x in range(new_width):
            old_x = int(x * scale_x)
            old_y = int(y * scale_y)

            old_x = min(old_x, old_width - 1)
            old_y = min(old_y, old_height - 1)

            resized_image[y, x] = image[old_y, old_x]
    
    return resized_image

if __name__ == "__main__":
    image = cv2.imread('test_image.jpg')

    if image is not None:
        print("Изображение успешно загружено!")
        print(f"Размер исходного изображения: {image.shape}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        resized = change_resolution(image_rgb, 300, 200)
        print(f"Размер после изменения: {resized.shape}")
        
        # Показываем результаты
        cv2.imshow('Original', image)  # Исходное (BGR)
        cv2.imshow('Resized', cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()