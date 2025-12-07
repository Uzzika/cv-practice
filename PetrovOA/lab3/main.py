import argparse
import os
import cv2
from classifier.bow_classifier import BOWClassifier
from classifier.vit_classifier import ViTClassifier

# Получаем директорию, где находится main.py
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def parse_args():
    """Парсинг аргументов командной строки"""
    # Создаем парсер, отключаем help по умолчанию, чтобы добавить его позже
    parser = argparse.ArgumentParser(description='Классификация изображений достопримечательностей', add_help=False)

    # Группа для основных режимов (обучение/тест)
    required_args = parser.add_argument_group('Обязательные аргументы для обучения/тестирования')
    required_args.add_argument('--data_dir', type=str,
                               help='Путь к директории с данными')
    required_args.add_argument('--train_file', type=str,
                               help='Путь к файлу со списком обучающих изображений')
    required_args.add_argument('--test_file', type=str,
                               help='Путь к файлу со списком тестовых изображений')

    # Группа для режима визуализации
    visualization_args = parser.add_argument_group('Аргументы для режима визуализации')
    visualization_args.add_argument('--visualize', type=str, default=None,
                                    help='Путь к изображению для визуализации SIFT-ключевых точек')
    visualization_args.add_argument('--visualize_output', type=str, default=None,
                                    help='Путь для сохранения изображения с визуализацией')
    visualization_args.add_argument('--visualize_style', type=str, choices=['rich', 'simple', 'not_scaled'],
                                    default='rich', help='Стиль отрисовки ключевых точек (rich/simple/not_scaled)')

    # Общие/опциональные аргументы
    optional_args = parser.add_argument_group('Общие/опциональные аргументы')
    optional_args.add_argument('--mode', type=str, choices=['train', 'test', 'both'],
                               default='both', help='Режим работы')
    optional_args.add_argument('--algorithm', type=str, choices=['bow', 'vit', 'both'],
                               default='both', help='Алгоритм классификации')
    optional_args.add_argument('--clusters', type=int, default=100,
                               help='Количество кластеров для метода мешок слов')
    optional_args.add_argument('--learning_rate', type=float, default=0.001,
                               help='Скорость обучения для ViT')
    optional_args.add_argument('--batch_size', type=int, default=16,
                               help='Размер батча для ViT')
    optional_args.add_argument('--epochs', type=int, default=20,
                               help='Количество эпох для ViT')
    optional_args.add_argument('--model_dir', type=str, default=None,
                               help='Директория для загрузки/сохранения модели (используется для BOW или ViT)')

    # Добавляем стандартный help
    optional_args.add_argument('-h', '--help', action='help', help='Показать это справочное сообщение и выйти')

    return parser.parse_args(), parser


def train_and_test_classifier(classifier, args):
    """Обучение и тестирование классификатора"""
    if args.mode in ['train', 'both']:
        # Загрузка обучающих данных
        train_paths, train_labels = classifier.load_data(args.train_file, args.data_dir)
        print(f"Загружено {len(train_paths)} обучающих изображений")

        # Обучение классификатора
        classifier.train(train_paths, train_labels)

    if args.mode in ['test', 'both']:
        # Загрузка тестовых данных
        test_paths, test_labels = classifier.load_data(args.test_file, args.data_dir)
        print(f"Загружено {len(test_paths)} тестовых изображений")

        # Тестирование классификатора
        predictions, accuracy = classifier.test(test_paths, test_labels)
        return accuracy
    return None

def visualize_sift_features(args):
    """Визуализация SIFT-ключевых точек на изображении"""
    if not os.path.exists(args.visualize):
        print(f"Ошибка: изображение не найдено: {args.visualize}")
        return

    print("\n=== Визуализация SIFT-ключевых точек ===")

    # Создаем классификатор для визуализации
    bow_classifier = BOWClassifier()

    # Визуализация
    result_image, num_keypoints, stats = bow_classifier.show_keypoints(
        args.visualize,
        output_path=args.visualize_output,
        draw_style=args.visualize_style
    )

    if result_image is not None:
        # Показываем изображение, если не указан путь для сохранения
        if args.visualize_output is None:
            print("\nНажмите любую клавишу для закрытия окна...")
            cv2.imshow('SIFT Keypoints Visualization', result_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

def main():
    """Основная функция"""
    args, parser = parse_args()

    # Если запущен режим визуализации, выполняем его и выходим
    if args.visualize:
        visualize_sift_features(args)
        return

    # Если это не режим визуализации, проверяем наличие обязательных аргументов
    if not all([args.data_dir, args.train_file, args.test_file]):
        parser.error("В режиме обучения/тестирования аргументы --data_dir, --train_file и --test_file являются обязательными.")

    results = {}

    # Обработка BOW классификатора
    if args.algorithm in ['bow', 'both']:
        print("\n=== Метод мешка слов (BOW) ===")
        # Если указан model_dir и используется только BOW, используем его, иначе используем bow_model
        if args.algorithm == 'bow' and args.model_dir is not None:
            bow_model_dir = args.model_dir
        else:
            # Используем абсолютный путь относительно директории скрипта
            bow_model_dir = os.path.join(SCRIPT_DIR, 'bow_model')
        bow_classifier = BOWClassifier(n_clusters=args.clusters,
                                     model_dir=bow_model_dir)
        bow_accuracy = train_and_test_classifier(bow_classifier, args)
        if bow_accuracy is not None:
            results['BOW'] = bow_accuracy

    # Обработка ViT классификатора
    if args.algorithm in ['vit', 'both']:
        print("\n=== Vision Transformer классификатор (ViT) ===")
        # Если указан model_dir и используется только ViT, используем его, иначе используем vit_model
        if args.algorithm == 'vit' and args.model_dir is not None:
            vit_model_dir = args.model_dir
        else:
            # Используем абсолютный путь относительно директории скрипта
            vit_model_dir = os.path.join(SCRIPT_DIR, 'vit_model')
        vit_classifier = ViTClassifier(
            model_dir=vit_model_dir,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            epochs=args.epochs
        )
        vit_accuracy = train_and_test_classifier(vit_classifier, args)
        if vit_accuracy is not None:
            results['ViT'] = vit_accuracy

    # Вывод сравнения результатов
    if results and args.mode in ['test', 'both']:
        print("\n=== Сравнение результатов ===")
        for method, accuracy in results.items():
            print(f"{method}: {accuracy:.3f}")


if __name__ == '__main__':
    main()
