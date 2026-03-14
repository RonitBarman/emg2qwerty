"""Plot training progress from TensorBoard logs in logs/<date>/<time>/lightning_logs/version_0."""

import glob
import os

import matplotlib
import matplotlib.pyplot as plt
from tbparse import SummaryReader

# --- Find the most recent run automatically ---
# Override LOG_DIR manually if you want a specific run, e.g.:
# LOG_DIR = "logs/2026-03-09/06-25-46/lightning_logs/version_0"
candidates = sorted(glob.glob("logs/*/*/lightning_logs/version_0"))
if not candidates:
    raise FileNotFoundError("No lightning_logs found under logs/")
LOG_DIR = candidates[-1]
print(f"Reading logs from: {LOG_DIR}")

OUTPUT_DIR = os.path.dirname(os.path.dirname(LOG_DIR))  # logs/<date>/<time>/

# --- Research paper style ---
matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.5,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

reader = SummaryReader(LOG_DIR)
df = reader.scalars

epoch_map = df[df["tag"] == "epoch"].drop_duplicates(subset="step", keep="last").set_index("step")["value"]

TRAIN_COLOR = "#7b2d8e"
VAL_COLOR = "#e8870e"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))

# --- (a) Loss vs Epoch ---
for tag, label, color in [
    ("train/loss_epoch", "Train", TRAIN_COLOR),
    ("val/loss", "Validation", VAL_COLOR),
]:
    sub = df[df["tag"] == tag].copy()
    if not sub.empty:
        sub["epoch"] = sub["step"].map(epoch_map).ffill()
        grouped = sub.groupby("epoch")["value"].mean()
        ax1.plot(grouped.index, grouped.values, label=label, color=color)
ax1.set_ylim(0, 4)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("(a) Loss", style="italic")
ax1.legend(frameon=True, fancybox=False, edgecolor="0.7")
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.tick_params(direction="in")

# --- (b) CER vs Epoch ---
for tag, label, color in [
    ("train/CER", "Train", TRAIN_COLOR),
    ("val/CER", "Validation", VAL_COLOR),
]:
    sub = df[df["tag"] == tag].copy()
    if not sub.empty:
        sub["epoch"] = sub["step"].map(epoch_map).ffill()
        grouped = sub.groupby("epoch")["value"].mean()
        ax2.plot(grouped.index, grouped.values, label=label, color=color)
ax2.set_ylim(0, 125)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("CER")
ax2.set_title("(b) Character Error Rate", style="italic")
ax2.legend(frameon=True, fancybox=False, edgecolor="0.7")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.tick_params(direction="in")

plt.tight_layout(w_pad=3.0)
out_png = os.path.join(OUTPUT_DIR, "training_progress.png")
out_pdf = os.path.join(OUTPUT_DIR, "training_progress.pdf")
plt.savefig(out_png)
plt.savefig(out_pdf)
plt.show()
print(f"Saved to {out_png} and {out_pdf}")
