# accuracy, precision, recall, F1, confusion matrix
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def classification_metrics(y_true, y_pred):
    """Macro averages weight every species equally (classes are ~balanced anyway)."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
    }


def top_k_accuracy(logits, y_true, k=5):
    """Fraction of samples whose true class is among the k highest logits."""
    top = logits.topk(k, dim=1).indices
    return (top == y_true.unsqueeze(1)).any(dim=1).float().mean().item()


def confusion(y_true, y_pred, num_classes):
    return confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
