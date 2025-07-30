import matplotlib.pyplot as plt
import seaborn as sns 
plt.rcParams.update({
    "text.usetex": False,                  # Use mathtext, not full LaTeX
    "font.family": "serif",               
    "font.serif": ["STIXGeneral"],         # or ["Computer Modern"], ["DejaVu Serif"]
    "mathtext.fontset": "stix",            # STIX mimics LaTeX math style
    "mathtext.rm": "serif",
    "mathtext.it": "serif:italic",
    "mathtext.bf": "serif:bold",
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "axes.unicode_minus": False,
})












def plot_metric(
    x,
    y_list,
    labels=None,
    x_label="Epoch",
    y_label="Value",
    title="Training Curve",
    seaborn_style="whitegrid",
    seaborn_palette="muted"
):
    """
    Plots one or more y-series against a common x-axis.

    Parameters:
        x (list or array): X-axis values (e.g., epochs)
        y_list (list of arrays): Multiple y-series to plot
        labels (list of str): Labels for legend
        x_label (str): Label for x-axis
        y_label (str): Shared label for y-axis
        title (str): Title of the plot
        seaborn_style (str): Seaborn style (e.g., whitegrid)
        seaborn_palette (str): Color palette (e.g., deep, muted)
    """
    sns.set_theme(style=seaborn_style, palette=seaborn_palette)
    plt.figure(figsize=(8, 5))

    if labels is None:
        labels = [f"Series {i+1}" for i in range(len(y_list))]

    for y, label in zip(y_list, labels):
        plt.plot(x, y, label=label)

    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)  # same y-label for all
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
