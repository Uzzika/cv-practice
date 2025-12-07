import os
from typing import List, Tuple


def parse_class_from_path(path: str) -> int:
    parts = path.replace("/", "\\").split("\\")
    for part in parts:
        if part.startswith("01_"):
            return 0
        elif part.startswith("04_"):
            return 1
        elif part.startswith("08_"):
            return 2

def load_split_by_ratio(train_list_file: str, data_root: str, train_ratio: float = 0.7) -> Tuple[List[str], List[str], List[int], List[int]]:
    with open(train_list_file, 'r', encoding='utf-8') as f:
        rel_paths = [line.strip() for line in f if line.strip()]

    abs_paths = []
    labels = []
    for rel_path in rel_paths:
        abs_path = os.path.join(data_root, rel_path)
        abs_paths.append(abs_path)
        labels.append(parse_class_from_path(rel_path))

    n_total = len(abs_paths)
    n_train = int(n_total * train_ratio)
    n_train = max(1, min(n_train, n_total - 1))

    train_paths = abs_paths[:n_train]
    train_labels = labels[:n_train]
    test_paths = abs_paths[n_train:]
    test_labels = labels[n_train:]

    print("Всего изображений: ", n_total)
    print(f"Train: {len(train_paths)} ({len(train_paths)/n_total:.0%})")
    print(f"Test: {len(test_paths)}  ({len(test_paths)/n_total:.0%})")

    class_names = ["NizhnyNovgorodKremlin", "ArkhangelskCathedral", "PalaceOfLabor"]
    print("Распределение по классам (train):")
    for cls_id, name in enumerate(class_names):
        count = train_labels.count(cls_id)
        print(f"  {name}: {count}")

    return train_paths, test_paths, train_labels, test_labels