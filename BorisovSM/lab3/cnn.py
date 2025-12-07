import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from PIL import Image

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torchmetrics.classification import MulticlassAccuracy

from classifier import Classifier


class ItemsDataset(Dataset):
    def __init__(self, items, class_to_idx, transform):
        self.items = items
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        y = self.class_to_idx[label]
        return img, y


class CNNClassifier(Classifier):
    def __init__(self, args):
        super().__init__(args)
        self.device = self._resolve_device()

    def _resolve_device(self) -> str:
        dev = getattr(self.args, "device", "cuda")
        if dev not in ("cpu", "cuda"):
            return "cuda" if torch.cuda.is_available() else "cpu"
        if dev == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return dev

    def _build_class_mapping(self, train_items):
        classes = sorted({lbl for _, lbl in train_items})
        return {c: i for i, c in enumerate(classes)}

    def _make_resnet50(self, num_classes, pretrained=True):
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model, weights

    def _stratified_split(self, items, test_size=0.2, seed=42):
        labels = [lbl for _, lbl in items]
        tr, va = train_test_split(
            items, test_size=test_size, stratify=labels, random_state=seed)
        return tr, va

    def _run_epoch(self, model, loader, criterion, optimizer=None, num_classes=2, train=True):
        if train:
            model.train()
        else:
            model.eval()

        metric_acc = MulticlassAccuracy(
            num_classes=num_classes).to(self.device)
        total_loss = 0.0
        total_samples = 0

        torch.set_grad_enabled(train)
        for imgs, ys in loader:
            imgs = imgs.to(self.device)
            ys = ys.to(self.device)

            if train:
                optimizer.zero_grad(set_to_none=True)

            logits = model(imgs)
            loss = criterion(logits, ys)

            if train:
                loss.backward()
                optimizer.step()

            preds = logits.argmax(dim=1)
            metric_acc.update(preds, ys)

            bs = imgs.size(0)
            total_loss += float(loss) * bs
            total_samples += bs

        torch.set_grad_enabled(True)

        mean_loss = total_loss / max(1, total_samples)
        acc = float(metric_acc.compute())
        return mean_loss, acc

    def train(self, train_items):
        args = self.args
        epochs = args.epochs
        lr = args.lr
        bs = args.batch_size
        val_size = args.val_size
        weights_out = args.model_out

        train_items, val_items = self._stratified_split(
            train_items, test_size=val_size)

        class_to_idx = self._build_class_mapping(train_items)
        idx_to_class = {v: k for k, v in class_to_idx.items()}

        model, weights = self._make_resnet50(
            num_classes=len(class_to_idx), pretrained=True)
        tf = weights.transforms()
        train_tf = tf
        val_tf = tf

        train_ds = ItemsDataset(train_items, class_to_idx, transform=train_tf)
        val_ds = ItemsDataset(val_items,   class_to_idx, transform=val_tf)

        train_loader = DataLoader(train_ds, batch_size=bs,
                                  shuffle=True,  num_workers=0)
        val_loader = DataLoader(val_ds,   batch_size=bs,
                                shuffle=False, num_workers=0)

        model = model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=1)

        best_acc = -1.0
        os.makedirs(os.path.dirname(weights_out) or ".", exist_ok=True)

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self._run_epoch(
                model,
                train_loader,
                criterion,
                optimizer=optimizer,
                num_classes=len(class_to_idx),
                train=True,
            )

            _, val_acc = self._run_epoch(
                model,
                val_loader,
                criterion,
                optimizer=None,
                num_classes=len(class_to_idx),
                train=False,
            )

            print(
                f"[Эпоха {epoch:02d}] train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  val_acc={val_acc:.4f}")

            scheduler.step(val_acc)

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "class_to_idx": class_to_idx,
                    "idx_to_class": idx_to_class,
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                    "epochs": epoch,
                }, weights_out)
                print(
                    f"Сохраняем лучший val_acc={val_acc:.4f} -> {weights_out}")

        print(f"Лучший val_acc={best_acc:.4f}")

    def test(self, test_items):
        weights_path = getattr(self.args, "model_in", None) or getattr(
            self.args, "model_out", "resnet50.pth"
        )

        ckpt = torch.load(weights_path, map_location=self.device)
        class_to_idx = ckpt["class_to_idx"]
        idx_to_class = ckpt["idx_to_class"]

        eval_weights = models.ResNet50_Weights.IMAGENET1K_V2
        test_tf = eval_weights.transforms()

        test_ds = ItemsDataset(test_items, class_to_idx, transform=test_tf)

        test_loader = DataLoader(test_ds, batch_size=32,
                                 shuffle=False, num_workers=0)

        model, _ = self._make_resnet50(
            num_classes=len(class_to_idx), pretrained=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(self.device)
        model.eval()

        y_true, y_pred = [], []
        with torch.no_grad():
            for imgs, ys in test_loader:
                imgs = imgs.to(self.device)
                ys = ys.to(self.device)
                logits = model(imgs)
                preds = logits.argmax(dim=1)
                y_true.extend(ys.cpu().tolist())
                y_pred.extend(preds.cpu().tolist())

        classes = [idx_to_class[i] for i in range(len(idx_to_class))]

        acc = accuracy_score(y_true, y_pred)
        report = classification_report(
            y_true, y_pred, target_names=classes, digits=4)
        cm = confusion_matrix(y_true, y_pred)

        print(f"Test accuracy: {acc:.4f}")
        print("Classification report:\n", report)
        print("Confusion matrix:\n", cm)

        self.last_report = (y_true, y_pred, classes, report, cm)
