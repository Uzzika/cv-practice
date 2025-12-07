import argparse
import os
import sys
from utils.dataset_loader import load_split_by_ratio
from utils.metrics import compute_and_show_metrics
from models.bovw_model import BoVWClassifier
from models.cnn_model import CNNClassifier


def main():
    parser = argparse.ArgumentParser(
        description="Классификация достопримечательностей Нижнего Новгорода",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Путь к корню данных (должен содержать ExtDataset и NNSUDataset)")
    parser.add_argument("--train_list", type=str, required=True,
                        help="Файл со списком всех изображений (относительные пути Windows-стиля)")
    parser.add_argument("--mode", choices=["train", "test", "train+test"], default="train+test",
                        help="Режим работы")
    parser.add_argument("--algo", choices=["bovw", "cnn"], required=True,
                        help="Алгоритм: 'bovw' или 'cnn'")
    
    parser.add_argument("--model_save", type=str, default="model.pth",
                        help="Путь для сохранения модели (.pkl для BoVW, .pth для CNN)")
                        
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Доля обучающей выборки (от 0.0 до 1.0). Остальное — тест.")

    # BoVW params
    parser.add_argument("--bovw_clusters", type=int, default=100,
                        help="Количество визуальных слов")
    parser.add_argument("--bovw_detector", type=str, default="SIFT",
                        choices=["SIFT", "ORB", "AKAZE"],
                        help="Детектор/дескриптор")

    # CNN params
    parser.add_argument("--cnn_model", type=str, default="resnet18",
                        choices=["resnet18", "mobilenet_v2"],
                        help="Архитектура CNN")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Число эпох")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Размер батча")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Скорость обучения")

    # Visualisation (BoVW only)
    parser.add_argument("--vis_kp", type=str, default=None,
                        help="Путь к изображению для визуализации ключевых точек (только при --algo bovw). "
                             "Пример: --vis_kp data/ExtDataset/01_NizhnyNovgorodKremlin/kremlin_1113075.jpg")
    parser.add_argument("--vis_save", type=str, default="keypoints.jpg",
                        help="Имя файла для сохранения изображения с ключевыми точками")

    args = parser.parse_args()

    train_paths, test_paths, train_labels, test_labels = load_split_by_ratio(train_list_file=args.train_list, data_root=args.data_dir, train_ratio=args.train_ratio)

    if args.algo == "bovw":
        model = BoVWClassifier(
            n_clusters=args.bovw_clusters,
            detector_name=args.bovw_detector
        )

        if "train" in args.mode:

            if args.vis_kp:
                print(f"\nВизуализация ДО обучения...")
                model.visualize_keypoints(args.vis_kp, save_to=f"before_{args.vis_save}")
                
            model.fit(train_paths, train_labels)
            model.save(args.model_save)
        if "test" in args.mode:
            if not model.is_fitted:
                print(f"Загрузка модели из {args.model_save}")
                model = BoVWClassifier.load(args.model_save)
            preds = model.predict(test_paths)
            class_names = ["NizhnyNovgorodKremlin", "ArkhangelskCathedral", "PalaceOfLabor"]
            _ = compute_and_show_metrics(test_labels, preds, class_names)
            if args.vis_kp:
                print("\nВизуализация загруженной модели...")
                model.visualize_keypoints(args.vis_kp, save_to=f"before_{args.vis_save}")

    elif args.algo == "cnn":
        if "train" in args.mode:
            print(f"\nИнициализация CNN: {args.cnn_model} (загрузка из models/{args.cnn_model}.php)")
            model = CNNClassifier(model_name=args.cnn_model, num_classes=3)
            model.fit(
                train_paths, train_labels,
                val_paths=test_paths, val_labels=test_labels,
                batch_size=args.batch_size,
                epochs=args.epochs,
                lr=args.lr
            )
            model.save(args.model_save)

        if "test" in args.mode:
            print(f"Загрузка модели из {args.model_save}")
            model = CNNClassifier.load(args.model_save)
            preds = model.predict(test_paths)
            class_names = ["NizhnyNovgorodKremlin", "ArkhangelskCathedral", "PalaceOfLabor       "]
            _ = compute_and_show_metrics(test_labels, preds, class_names)

    else:
        raise ValueError(f"Неизвестный алгоритм: {args.algo}")


if __name__ == "__main__":
    main()