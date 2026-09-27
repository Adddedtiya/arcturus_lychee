from typing import Optional
from sklearn.metrics import classification_report, confusion_matrix


def generate_report(
        y_true      : list[int],
        y_pred      : list[int],
        class_names : Optional[list[str]] = None,
    ) -> list[str]:
    """Return the scikit-learn classification report as text lines.

    If class_names is given, the report always has one row for each class.
    This is also true for a class that is not in y_true or y_pred. Without
    the explicit labels, scikit-learn raises an error for this case, or gives
    rows with the wrong names.
    """
    if class_names is not None:
        full_report = classification_report(
            y_true, y_pred,
            labels        = list(range(len(class_names))),
            target_names  = class_names,
            zero_division = 0,
        )
    else:
        full_report = classification_report(y_true, y_pred, zero_division = 0)

    return full_report.splitlines()


def generate_confusion_matrix(
        y_true      : list[int],
        y_pred      : list[int],
        class_names : Optional[list[str]] = None,
    ) -> list[str]:
    """Return the confusion matrix as text lines. Rows are the true classes. Columns are the predictions.

    If class_names is given, the matrix is always N x N, with N classes.
    Without class_names, the matrix has the class indices that occur in
    y_true or y_pred.
    """
    if class_names is None:
        labels         = sorted(set(y_true + y_pred))
        display_labels = labels
    else:
        labels         = list(range(len(class_names)))
        display_labels = class_names

    suffix = "Actual \\ Prediction"
    matrix = confusion_matrix(y_true, y_pred, labels = labels)

    # The width of each column: the longest label or value, plus 2 spaces.
    max_label_len = max(len(str(label)) for label in display_labels) if display_labels else 1
    max_value_len = len(str(matrix.max())) if matrix.size else 1
    padding       = max(max_label_len, max_value_len, len(suffix)) + 2

    header = suffix.ljust(padding + 1) + "| " + "".join(str(label).ljust(padding) for label in display_labels)
    lines  = [header, "-" * len(header)]

    for i, label in enumerate(display_labels):
        row_values = "".join(str(count).ljust(padding) for count in matrix[i])
        lines.append(f"{str(label).ljust(padding)} | {row_values}")

    return lines
