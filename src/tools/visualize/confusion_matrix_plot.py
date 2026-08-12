import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

from utils import get_next_filename


def plot_confusion_matrix(labels, preds):

    cm = confusion_matrix(labels, preds)

    plt.figure(figsize=(6,6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.title("Confusion Matrix")

    path = get_next_filename("fig/confusion", "confusion")

    plt.savefig(path)
    plt.close()

    print("Saved:", path)
