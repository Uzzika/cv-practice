import argparse
from data_reader import get_train_test
from classifier import Classifier


def cli_argument_parser():
    parser = argparse.ArgumentParser(description="Classification")

    parser.add_argument("--data_dir", required=True, type=str,
                        help="Корневая директория с ExtDataset/ и NNSUDataset/")
    parser.add_argument("--train_list", required=True, type=str,
                        help="Файл со списком путей тренировочной выборки")
    parser.add_argument("--mode", choices=["train", "test", "both"], default="both",
                        help="train: обучение; test: тест по загруженной модели; both: обучение+тест")
    parser.add_argument("--algo", choices=["bow", "cnn"], default="bow",
                        help="Какой алгоритм использовать: мешок слов (bow) или ResNet50 (cnn)")

    parser.add_argument("--k", type=int, default=300,
                        help="Размер словаря BoW (KMeans)")
    parser.add_argument("--C", type=float, default=1.0,
                        help="Параметр C для LinearSVC")
    parser.add_argument("--max_kp", type=int, default=1000,
                        help="Макс. число ключевых точек (SIFT)")

    parser.add_argument("--vizualize_dir", type=str,
                        help="Путь к сохранению изображений с ключевыми точками SIFT")
    parser.add_argument("--viz_per_class", type=int, default=2,
                        help="Сколько изображений на класс визуализировать (SIFT)")

    parser.add_argument("--val_size", type=float, default=0.2,
                        help="Доля валидации, выделяемая из train (CNN)")
    parser.add_argument("--epochs", type=int, default=8,
                        help="Эпохи обучения CNN")
    parser.add_argument("--batch_size", type=int,
                        default=16, help="Batch size для CNN")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate для CNN")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Устройство для CNN: cuda или cpu")

    parser.add_argument("--model_in", type=str,
                        help="Путь к сохранённой модели")
    parser.add_argument("--model_out", type=str,
                        help="Куда сохранить обученную модель")

    return parser.parse_args()


def main():
    args = cli_argument_parser()

    train_items, test_items = get_train_test(args.data_dir, args.train_list)
    print(f"Размер тренировочной выборки: {len(train_items)}")
    print(f"Размер тестовой выборки: {len(test_items)}")

    clf = Classifier.create(args)
    if args.mode == "train":
        clf.train(train_items)
    elif args.mode == "test":
        clf.test(test_items)
    else:
        clf.train(train_items)
        clf.test(test_items)


if __name__ == "__main__":
    main()
