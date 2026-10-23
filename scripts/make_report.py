import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--name",type=str,required=True)
args = parser.parse_args()
from cifar_resnet.evaluate import confusion_matrix,per_class_metrics,calibration
d = np.load(f"artifacts/preds/{args.name}.npz")
out_dir = f"artifacts/report/{args.name}"
os.makedirs(out_dir, exist_ok=True)
probs = d["probs"]
preds = probs.argmax(axis=1)
labels = d["labels"]
C = confusion_matrix(preds, labels)
ece, counts, accs, confs = calibration(probs=probs,labels=labels,n_bins=15)
# Bin centres and non-empty bins
bins = np.linspace(0, 1, 16)
centres = (bins[:-1] + bins[1:]) / 2
nonempty = counts > 0

# Create the reliability diagram
plt.figure(figsize=(8, 6))

# Accuracy bars
plt.bar(
    centres[nonempty],
    accs[nonempty],
    width=1/15,
    alpha=0.6,
    label="Accuracy"
)

# Display counts above each bar
for x, acc, count in zip(
    centres[nonempty],
    accs[nonempty],
    counts[nonempty]
):
    plt.text(
        x, acc + 0.015, str(count),
        ha="center",
        va="bottom",
        fontsize=8
    )

# Perfect calibration diagonal
plt.plot(
    [0, 1], [0, 1],
    linestyle="--",
    color="black",
    label="Perfect calibration"
)

# Average confidence markers
plt.plot(
    centres[nonempty],
    confs[nonempty],
    marker="o",
    linestyle="none",
    label="Mean confidence"
)

# Axes and title
plt.xlim(0, 1)
plt.ylim(0, 1.1)  # Leave room for count labels
plt.xlabel("confidence")
plt.ylabel("accuracy")
plt.title(f"Reliability Diagram (ECE = {ece:.4f})")
plt.legend()
plt.tight_layout()

# Save and close without opening a window
plt.savefig(f"{out_dir}/reliability.png", dpi=300)
plt.close()

# Load the raw CIFAR-10 test images
import torchvision

test_dataset = torchvision.datasets.CIFAR10(
    root="data",
    train=False,
    download=True
)

class_names = test_dataset.classes

# Predicted classes and confidence scores
preds = probs.argmax(axis=1)
confidence = probs.max(axis=1)

# Find incorrect predictions
wrong_indices = np.where(preds != labels)[0]

# Sort by confidence, highest first, and take the top 20
sorted_wrong = wrong_indices[
    np.argsort(confidence[wrong_indices])[::-1]
]
top20 = sorted_wrong[:20]

# Print the 20 most confident errors
print("\nTop 20 most confident errors:")
for idx in top20:
    print(
        f"index: {idx:5d} | "
        f"pred: {class_names[preds[idx]]:10s} | "
        f"true: {class_names[labels[idx]]:10s} | "
        f"confidence: {confidence[idx]:.2f}"
    )

# Plot the raw images in a 4 x 5 grid
fig, axes = plt.subplots(4, 5, figsize=(14, 11))

for ax in axes.flat:
    ax.axis("off")

for ax, idx in zip(axes.flat, top20):
    image, _ = test_dataset[int(idx)]

    ax.imshow(np.asarray(image))
    ax.set_title(
        f"pred: {class_names[preds[idx]]} "
        f"{confidence[idx]:.2f} / "
        f"true: {class_names[labels[idx]]}",
        fontsize=9
    )
    ax.axis("off")

fig.tight_layout()

# Save and close
fig.savefig(
    f"{out_dir}/confident_errors.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)


# Class names
class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Predictions and confusion matrix
preds = probs.argmax(axis=1)
C = confusion_matrix(preds, labels)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
plt.imshow(C, cmap="Blues")
plt.xticks(range(10), class_names, rotation=45, ha="right")
plt.yticks(range(10), class_names)
plt.xlabel("predicted")
plt.ylabel("true")
plt.colorbar()

# Write counts in each cell with readable text color
threshold = C.max() / 2

for i in range(10):
    for j in range(10):
        color = "white" if C[i, j] > threshold else "black"
        plt.text(
            j, i, str(C[i, j]),
            ha="center",
            va="center",
            color=color
        )

plt.tight_layout()
plt.savefig(f"{out_dir}/confusion.png", dpi=300)
plt.close()

precision, recall, F1 = per_class_metrics(C)

accuracy = np.trace(C) / C.sum()

lines = [
    f"# Evaluation report: {args.name}",
    "",
    "| class | precision | recall | F1 |",
    "|---|---|---|---|",
]
for name, p, r, f in zip(class_names, precision, recall, F1):
    lines.append(f"| {name} | {p:.3f} | {r:.3f} | {f:.3f} |")

lines += [
    "",
    f"- Accuracy: {accuracy:.4f}",
    f"- Mean F1: {F1.mean():.4f}",
    f"- ECE (15 bins): {ece:.4f}",
]

with open(f"{out_dir}/metrics.md", "w") as f:
    f.write("\n".join(lines))