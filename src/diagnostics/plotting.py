"""Headless plotting helpers shared by experiments."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def loss_plot(history, path, title):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["epoch"], history["Train"], label="Train")
    ax.plot(history["epoch"], history["Validation"], label="Validation")
    ax.set(xlabel="Epoch", ylabel="Loss", title=title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
