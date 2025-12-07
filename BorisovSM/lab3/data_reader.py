from pathlib import Path

CLASS_DIRS = {
    "01_NizhnyNovgorodKremlin": "Kremlin",
    "04_ArkhangelskCathedral": "ArkhangelskCathedral",
    "08_PalaceOfLabor": "PalaceOfLabor",
}


def label_of(rel):
    for part in Path(rel).parts:
        if part in CLASS_DIRS:
            return CLASS_DIRS[part]
    return Path(rel).parent.name


def get_train_test(data_root: str, train_txt: str):
    root = Path(data_root)

    train_rels = set()
    with open(train_txt, "r", encoding="utf-8") as f:
        for line in f:
            rel = line.strip()
            if rel:
                train_rels.add(rel)

    all_rels = set()
    for p in root.rglob("*.jpg"):
        if p.is_file() and p.suffix.lower() == ".jpg":
            all_rels.add(str(p.relative_to(root)))

    test_rels = sorted(all_rels - train_rels)

    train_items = [(str(root / rel), label_of(rel))
                   for rel in sorted(train_rels)]
    test_items = [(str(root / rel), label_of(rel)) for rel in test_rels]
    return train_items, test_items
