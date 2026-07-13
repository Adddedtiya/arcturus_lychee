from typing import Union
from sklearn.metrics import classification_report, confusion_matrix

def generate_report(x_true : list[int], y_pred : list[int], class_names : Union[list[str], None] = None) -> list[str]:

    # Pass explicit integer labels when class_names are supplied so the report
    # stays aligned even if some classes never appear in y_true / y_pred (which
    # otherwise makes sklearn raise, or silently mismatch target_names).
    if class_names is not None:
        labels = list(range(len(class_names)))
        full_report = classification_report(
            x_true, y_pred,
            labels       = labels,
            target_names = class_names,
            zero_division = 0,
        )
    else:
        full_report = classification_report(x_true, y_pred, zero_division = 0)

    return full_report.splitlines()

def generate_confusion_matrix(x_true : list[int], y_pred : list[int], class_names : Union[list[str], None] = None) -> list[str]:

    # 1. Determine the label set + how they are displayed
    if class_names is None:
        labels         = sorted(list(set(x_true + y_pred)))
        display_labels = labels
    else:
        # fixed, full label set (0..N-1) so the matrix is always N x N and lines
        # up with class_names regardless of which classes are present
        labels         = list(range(len(class_names)))
        display_labels = class_names

    # Header message
    suffix = "Actual \\ Prediction"

    # 2. Generate the raw matrix over the fixed label set
    cm = confusion_matrix(x_true, y_pred, labels = labels)

    # 3. Calculate dynamic padding
    max_label_len = max([len(str(l)) for l in display_labels]) if display_labels else 1
    max_val_len   = len(str(cm.max())) if cm.size else 1
    padding = max(max_label_len, max_val_len, len(suffix)) + 2  # +2 breathing room

    # 4. Build the string
    report_lines = []

    header = suffix.ljust(padding + 1) + "| " + "".join([f"{str(l).ljust(padding)}" for l in display_labels])
    report_lines.append(header)
    report_lines.append("-" * len(header))

    for i, label in enumerate(display_labels):
        row_values = "".join([str(count).ljust(padding) for count in cm[i]])
        report_lines.append(f"{str(label).ljust(padding)} | {row_values}")

    return report_lines
