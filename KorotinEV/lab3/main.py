import os
import argparse
import cv2
import numpy as np
from bow_classifier import BOWClassifier
from cnn_classifier import CNNClassifier

def create_classifier(algorithm, **kwargs):
    if algorithm == 'bow':
        return BOWClassifier(
            n_clusters=kwargs.get('n_clusters', 100),
            image_size=kwargs.get('image_size', (224, 224)),
            class_names=kwargs.get('class_names')
        )
    elif algorithm == 'cnn':
        from cnn_classifier import CNNClassifier
        return CNNClassifier(
            image_size=kwargs.get('image_size', (224, 224)),
            batch_size=kwargs.get('batch_size', 16),
            epochs=kwargs.get('epochs', 5),
            learning_rate=kwargs.get('learning_rate', 0.001),
            dropout_rate=kwargs.get('dropout_rate', 0.5),
            class_names=kwargs.get('class_names')
        )
    else:
        raise ValueError(f"Неизвестный алгоритм: {algorithm}")

def visualize_sift_features(image_path, output_path=None):
    classifier = BOWClassifier()
    
    result_image = classifier.visualize_sift(image_path, output_path)
    
    if result_image is not None and output_path is None:
        cv2.imshow('SIFT Features', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return result_image

def parser():
    parser = argparse.ArgumentParser(description='Классификация достопримечательностей Нижнего Новгорода')
    parser.add_argument('--data_dir', type=str, default='./NNClassification', help='Путь к директории с изображениями')
    parser.add_argument('--train_file', type=str, default='train_test_split/train.txt', help='Путь к файлу train.txt')
    parser.add_argument('--test_file', type=str, default='train_test_split/test.txt', help='Путь к файлу test.txt')
    parser.add_argument('--algorithm', type=str, choices=['bow', 'cnn'], default='bow',
                       help='Алгоритм классификации: bow (мешок слов) или cnn (нейронная сеть)')
    parser.add_argument('--clusters', type=int, default=100, help='Количество кластеров для метода мешок слов')
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'both', 'visualize'], default='visualize',
                       help='Режим работы: train (обучение), test (тестирование), both (обучение и тестирование), visualize (визуализация SIFT)')
    parser.add_argument('--model_dir', type=str, default='models', help='Директория для сохранения/загрузки моделей')
    
    parser.add_argument('--batch_size', type=int, default=16, help='Размер батча для CNN')
    parser.add_argument('--epochs', type=int, default=5, help='Количество эпох для CNN')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Скорость обучения для CNN')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate для CNN')
    parser.add_argument('--image_size', type=int, nargs=2, default=[224, 224], 
                       help='Размер изображения (ширина высота)')
    
    parser.add_argument('--image_path', default='./NNClassification/ExtDataset/01_NizhnyNovgorodKremlin/7377598dba3c816bcda655d07bd4695d.jpeg', type=str, help='Путь к изображению для визуализации SIFT дескрипторов')
    parser.add_argument('--output_path', type=str, help='Путь для сохранения изображения с SIFT дескрипторами')

    return parser.parse_args()

def main():
    args = parser()
    
    if args.mode == 'visualize':
        if not args.image_path:
            print("Для режима visualize требуется указать --image_path")
            return
        
        if not os.path.exists(args.image_path):
            print(f"Изображение не найдено: {args.image_path}")
            return
            
        print(f"Визуализация SIFT дескрипторов для изображения: {args.image_path}")
        visualize_sift_features(args.image_path, args.output_path)
        return
    
    if args.mode in ['train', 'both'] and not args.train_file:
        parser.error("Для режима обучения требуется указать --train_file")
    
    if args.mode in ['test', 'both'] and not args.test_file:
        parser.error("Для режима тестирования требуется указать --test_file")
    
    classifier_kwargs = {
        'n_clusters': args.clusters,
        'image_size': tuple(args.image_size),
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'dropout_rate': args.dropout_rate
    }
    
    classifier = create_classifier(args.algorithm, **classifier_kwargs)

    print(f"Режим работы: {args.mode}")
    print(f"Алгоритм: {args.algorithm}")
    print(f"Директория с данными: {args.data_dir}")
    
    train_accuracy = 0
    test_accuracy = 0
    
    if args.mode in ['train', 'both']:
        train_accuracy = classifier.train(args.train_file, args.data_dir)
        classifier.save_model(args.model_dir)
    
    if args.mode in ['test', 'both']:
        if args.mode == 'test':
            if not os.path.exists(args.model_dir):
                print(f"Ошибка: директория с моделью {args.model_dir} не существует!")
                return
            classifier.load_model(args.model_dir)
        
        test_accuracy = classifier.test(args.test_file, args.data_dir)
    
    print("\n" + "="*50)
    print("РЕЗУЛЬТАТЫ:")
    print(f"Алгоритм: {args.algorithm}")
    print(f"Режим работы: {args.mode}")
    
    if args.mode in ['train', 'both']:
        print(f"Accuracy на обучающей выборке: {train_accuracy:.4f}")
    
    if args.mode in ['test', 'both']:
        print(f"Accuracy на тестовой выборке: {test_accuracy:.4f}")
    
    print("="*50)

if __name__ == "__main__":
    main()
