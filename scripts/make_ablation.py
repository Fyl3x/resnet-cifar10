import numpy as np 
import os
import csv 
def load_log(name):
    path = f"artifacts/logs/{name}.csv"
    if not os.path.exists(path=path):
        return None
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        iterations = []
        train_errors = []
        val_errors = []
        for row in reader:
            train_errors.append(100 * (1 - float(row["train_acc"])))
            val_errors.append(100 * (1 - float(row["val_acc"])))
            iterations.append(int(row["iteration"]))
    return (
    np.array(iterations),
    np.array(train_errors),
    np.array(val_errors)
)


names = ["dev_res20","abl_plain20","abl_res56","abl_plain56"]

for name in names:
    result = load_log(name=name)
    if result is None:
        continue

    iterations, train_errors, val_errors = result
    print(
    f"{name}: rows={len(iterations)}, "
    f"last train error={train_errors[-1]:.2f}%, "
    f"last validation error={val_errors[-1]:.2f}%"
)


import matplotlib.pyplot as plt

# Run names and colours by depth
plain_runs = [
    ("abl_plain20", "Plain-20", "tab:blue"),
    ("abl_plain56", "Plain-56", "tab:orange"),
]

residual_runs = [
    ("dev_res20", "ResNet-20", "tab:blue"),
    ("abl_res56", "ResNet-56", "tab:orange"),
]

# Two panels sharing the y-axis
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

for ax, title, runs in [
    (axes[0], "plain", plain_runs),
    (axes[1], "residual", residual_runs),
]:
    for name, label, color in runs:
        result = load_log(name)

        if result is None:
            continue

        iterations, train_errors, val_errors = result

        # Thin, faint training curve
        ax.plot(
            iterations,
            train_errors,
            color=color,
            alpha=0.35,
            linewidth=1,
            label=f"{label} train",
        )

        # Bold validation curve
        ax.plot(
            iterations,
            val_errors,
            color=color,
            linewidth=2,
            label=f"{label} validation",
        )

    # Learning-rate drops
    ax.axvline(32000, color="gray", linestyle=":", linewidth=1)
    ax.axvline(48000, color="gray", linestyle=":", linewidth=1)

    ax.set_title(title)
    ax.set_xlabel("iteration")
    ax.set_ylim(0, 20)
    ax.grid(True, alpha=0.2)
    ax.legend()

axes[0].set_ylabel("error (%)")

# Save the figure
out_dir = "artifacts/report/ablation"
os.makedirs(out_dir, exist_ok=True)

fig.tight_layout()
fig.savefig(f"{out_dir}/curves.png", dpi=300)
plt.close()


# Ablkation Table 


runs = [
    ("dev_res20", 20, "residual"),
    ("abl_plain20", 20, "plain"),
    ("abl_res56", 56, "residual"),
    ("abl_plain56", 56, "plain"),
]

rows = []

for name, depth, kind in runs:
    result = load_log(name)
    if result is None or len(result[0]) == 0:
        train_error = "pending"
        val_error = "pending"
    else:
        iterations, train_errors, val_errors = result
        train_error = f"{train_errors[-1]:.2f}%"
        val_error = f"{val_errors[-1]:.2f}%"
    pred_path = f"artifacts/preds/{name}.npz"

    if os.path.exists(pred_path):
        d = np.load(pred_path)
        probs = d["probs"]
        labels = d["labels"]    
        preds = probs.argmax(axis=1)
        test_error = f"{100 * (1 - np.mean(preds == labels)):.2f}%"
    else:
        test_error = "pending"
    rows.append((name, depth, kind, train_error, val_error, test_error))


os.makedirs("artifacts/report/ablation", exist_ok=True)

table_path = "artifacts/report/ablation/table.md"

with open(table_path, "w", encoding="utf-8") as f:
    f.write("| Run | Depth | Type | Train error | Validation error | Test error |\n")
    f.write("|---|---:|---|---:|---:|---:|\n")

    for name, depth, kind, train_error, val_error, test_error in rows:
        f.write(
            f"| {name} | {depth} | {kind} | "
            f"{train_error} | {val_error} | {test_error} |\n"
        )

print(f"Saved table to {table_path}")