import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from hiclass.metrics import f1, precision, recall
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder


def show_confusion_matrix(
    y_true, y_pred, label_encoder, figsize=(8, 8), title_suffix=""
):
    """
    Display a labeled confusion matrix heatmap with counts and row-wise percentages.

    This function generates and displays a confusion matrix using seaborn's heatmap.
    Each cell is annotated with the raw count and the percentage of that row (i.e.,
    normalized over the true labels). Labels are decoded using the provided LabelEncoder.

    Parameters:
    ----------
    y_true : array-like
        True class labels (as integer-encoded values).

    y_pred : array-like
        Predicted class labels (as integer-encoded values).

    label_encoder : sklearn.preprocessing.LabelEncoder
        A fitted LabelEncoder used to convert class indices back to their string labels
        for axis tick labels.

    figsize : tuple, default=(8, 8)
        Size of the matplotlib figure.

    title_suffix : str, optional
        Optional string to append to the plot title for distinguishing between different
        confusion matrices (e.g., per level or category).

    Output:
    ------
    Displays a confusion matrix heatmap using matplotlib with annotations showing:
      - The number of samples per cell (count)
      - The percentage of the row (true label) that each prediction represents

    Notes:
    -----
    - Percentages are normalized across rows (i.e., per true label).
    - Uses seaborn for visualization and matplotlib for display.
    - Handles division by zero using np.nan_to_num.
    """
    cm = confusion_matrix(y_true, y_pred)
    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    # Normalizes percentages over true labels (rows)
    group_percentages = [
        "{0:.2%}".format(value)
        for value in np.nan_to_num(cm / cm.sum(axis=1, keepdims=True)).flatten()
    ]
    annotations = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts, group_percentages)]
    annotations = np.asarray(annotations).reshape(cm.shape)
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=annotations,
        fmt="",
        cmap="Blues",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_,
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    if title_suffix:
        plt.title(f"Confusion Matrix - {title_suffix}")
    else:
        plt.title("Confusion Matrix")
    plt.show()


def evaluation_report(y_true, y_pred, show_confusion_matrices=True):
    """
    Generate a detailed evaluation report for hierarchical classification results.

    This function evaluates a three-level hierarchical classification by computing
    and printing the precision, recall, and F1-score at each level (Level 1, Level 2,
    and Level 3). Since these are computed for each level separately, there's no need
    for hierarchical variants of these scores. It also prints overall micro and macro
    averaged scores using the full hierarchical labels. Optionally, confusion matrices
    can be displayed per level.

    Parameters:
    ----------
    y_true : list of tuples
        True labels for each instance. Each label is a tuple representing the hierarchy
        (e.g., (level1_label, level2_label, level3_label)). The number of levels may vary
        between instances.

    y_pred : list of tuples
        Predicted labels for each instance, structured the same way as `y_true`.

    show_confusion_matrices : bool, default=True
        Whether to display confusion matrices for each classification level. Level 2
        matrices are shown per Level 1 label, and Level 3 matrices per Level 2 label.

    Output:
    ------
    Prints precision, recall, and F1-score (macro and micro averaged) for:
      - the full hierarchical labels,
      - each individual level (Level 1, 2, and 3),
    and optionally shows confusion matrices per level and sub-level.

    Notes:
    -----
    - Level 2 metrics are only computed for instances where Level 1 matches.
    - Level 3 metrics are only computed for instances where both Level 1 and 2 match.
    - Uses LabelEncoder internally to convert labels to integers for sklearn metrics.
    - Assumes the existence of helper functions `f1`, `precision`, `recall`, and
      `show_confusion_matrix`.

    """
    y_true_l1 = [label[0] for label in y_true]
    y_pred_l1 = [label[0] for label in y_pred]
    le_l1 = LabelEncoder()
    le_l1.fit(y_true_l1 + y_pred_l1)
    y_true_l1 = le_l1.transform(y_true_l1)
    y_pred_l1 = le_l1.transform(y_pred_l1)

    y_true_l2 = [
        true[1]
        for true, pred in zip(y_true, y_pred)
        if true[0] == pred[0] and len(true) > 1
    ]
    y_pred_l2 = [
        pred[1]
        for true, pred in zip(y_true, y_pred)
        if true[0] == pred[0] and len(true) > 1
    ]
    le_l2 = LabelEncoder()
    le_l2.fit(y_true_l2 + y_pred_l2)
    y_true_l2 = le_l2.transform(y_true_l2)
    y_pred_l2 = le_l2.transform(y_pred_l2)

    y_true_l3 = [
        true[2]
        for true, pred in zip(y_true, y_pred)
        if all(true[:2] == pred[:2]) and len(true) > 2
    ]
    y_pred_l3 = [
        pred[2]
        for true, pred in zip(y_true, y_pred)
        if all(true[:2] == pred[:2]) and len(true) > 2
    ]
    le_l3 = LabelEncoder()
    le_l3.fit(y_true_l3 + y_pred_l3)
    y_true_l3 = le_l3.transform(y_true_l3)
    y_pred_l3 = le_l3.transform(y_pred_l3)

    for average in ["macro", "micro"]:
        print(f"Overall {average}:")
        print(f"\tF1 {average}: {f1(y_true, y_pred, average=average):.3f}")
        print(
            f"\tPrecision {average}: {precision(y_true, y_pred, average=average):.3f}"
        )
        print(f"\tRecall {average}: {recall(y_true, y_pred, average=average):.3f}")

    print("\nLevel 1:")
    for average in ["macro", "micro"]:
        print(f"\t{average.capitalize()}:")
        print(
            f"\t\tF1 {average}: {f1_score(y_true_l1, y_pred_l1, average=average):.3f}"
        )
        print(
            f"\t\tPrecision {average}: {precision_score(y_true_l1, y_pred_l1, average=average):.3f}"
        )
        print(
            f"\t\tRecall {average}: {recall_score(y_true_l1, y_pred_l1, average=average):.3f}"
        )

    if show_confusion_matrices:
        show_confusion_matrix(y_true_l1, y_pred_l1, le_l1, title_suffix="Level 1")

    print("\nLevel 2:")
    for average in ["macro", "micro"]:
        print(f"\t{average.capitalize()}:")
        print(
            f"\t\tF1 {average}: {f1_score(y_true_l2, y_pred_l2, average=average):.3f}"
        )
        print(
            f"\t\tPrecision {average}: {precision_score(y_true_l2, y_pred_l2, average=average):.3f}"
        )
        print(
            f"\t\tRecall {average}: {recall_score(y_true_l2, y_pred_l2, average=average):.3f}"
        )

    if show_confusion_matrices:
        for label in le_l1.classes_:
            y_true_temp = [
                x
                for x in le_l2.inverse_transform(y_true_l2)
                if str(x).startswith(label)
            ]
            y_pred_temp = [
                x
                for x in le_l2.inverse_transform(y_pred_l2)
                if str(x).startswith(label)
            ]

            if len(y_true_temp) == 0:
                continue
            else:
                le_l2_temp = LabelEncoder()
                le_l2_temp.fit(y_true_temp + y_pred_temp)
                show_confusion_matrix(
                    y_true_temp, y_pred_temp, le_l2_temp, title_suffix=label
                )

    print("\nLevel 3:")
    for average in ["macro", "micro"]:
        print(f"\t{average.capitalize()}:")
        print(
            f"\t\tF1 {average}: {f1_score(y_true_l3, y_pred_l3, average=average):.3f}"
        )
        print(
            f"\t\tPrecision {average}: {precision_score(y_true_l3, y_pred_l3, average=average):.3f}"
        )
        print(
            f"\t\tRecall {average}: {recall_score(y_true_l3, y_pred_l3, average=average):.3f}"
        )

    if show_confusion_matrices:
        for label in le_l2.classes_:
            y_true_temp = [
                x
                for x in le_l3.inverse_transform(y_true_l3)
                if str(x).startswith(label)
            ]
            y_pred_temp = [
                x
                for x in le_l3.inverse_transform(y_pred_l3)
                if str(x).startswith(label)
            ]

            if len(y_true_temp) == 0:
                continue
            else:
                le_l2_temp = LabelEncoder()
                le_l2_temp.fit(y_true_temp + y_pred_temp)
                show_confusion_matrix(
                    y_true_temp, y_pred_temp, le_l2_temp, title_suffix=label
                )
