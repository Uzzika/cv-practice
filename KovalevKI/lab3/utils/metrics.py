import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os


def compute_and_show_metrics(y_true: list, y_pred: list, class_names: list = None, save_path: str = None):
    if class_names is None:
        class_names = [f"Class {i}" for i in range(max(max(y_true), max(y_pred)) + 1)]
    
    labels = list(range(len(class_names)))

    accuracy = np.mean(np.array(y_true) == np.array(y_pred))
    print(f"\nТочность (accuracy): {accuracy:.4f} ({accuracy * 100:.2f}%)")

    print("\nПодробный отчёт:")
    print(classification_report(
        y_true, y_pred, 
        labels=labels,
        target_names=class_names, 
        digits=4,
        zero_division=0
    ))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("\nConfusion matrix:")
    header = "                     " + "  ".join([f"{name:>8}" for name in class_names])
    print(header)
    for i, row in enumerate(cm):
        row_str = f"{class_names[i]:<6}" + "                 ".join([f"{val:8d}" for val in row])
        print(row_str)
    try:
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Матрица рассогласования')
        plt.ylabel('Истинный класс')
        plt.xlabel('Предсказанный класс')
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nМатрица сохранена: {os.path.abspath(save_path)}")
        plt.close()
    except Exception as e:
        print(f"\nНе удалось построить график: {e}")

    return accuracy