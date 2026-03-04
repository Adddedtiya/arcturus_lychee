import numpy  as np
import pandas as pd

from typing import Union
from sklearn.metrics import classification_report, confusion_matrix

def generate_report(x_true : list[int], y_pred : list[int], class_names : Union[list[str], None] = None) -> list[str]:
    full_report = classification_report(x_true, y_pred, target_names = class_names)
    report_list = full_report.splitlines()
    return report_list

def generate_confusion_matrix(x_true : list[int], y_pred : list[int], class_names : Union[list[str], None] = None) -> list[str]:

    # 1. Determine the unique labels
    if class_names is None:
        labels = sorted(list(set(x_true + y_pred)))
    else:
        labels = class_names

    # Header message
    suffix = "Actual \ Prediction" 

    # 2. Generate the raw matrix
    cm = confusion_matrix(x_true, y_pred, labels=labels)

    # 3. Calculate dynamic padding
    # We want the cell width to be the max of (longest label length) or (longest number length)
    max_label_len = max([len(str(l)) for l in labels])
    max_val_len = len(str(cm.max()))
    padding = max(max_label_len, max_val_len, len(suffix)) + 2  # +2 for some breathing room
    
    # 4. Build the String
    report_lines = []
    
    # Header Row (Predicted labels)
    header = suffix.ljust(padding) + "".join([f"{str(l).ljust(padding)}" for l in labels])
    report_lines.append(header)
    report_lines.append("-" * len(header)) # Separator line
    
    # Data Rows
    for i, label in enumerate(labels):
        row_values = "".join([str(count).ljust(padding) for count in cm[i]])
        report_lines.append(f"{str(label).ljust(padding)} | {row_values}")

    return report_lines