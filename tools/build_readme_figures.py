"""Verify the original baseline run and render the figures used in the README.

Example (from the repository root):
    python tools/build_readme_figures.py --data-dir /path/to/NEU-CLS --run-dir /path/to/run_001
The original best_model.pth and experiment records must be present in run-dir.
"""
import argparse
import csv
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_split import IDX_TO_CLASS, load_split_manifest
from dataset import create_datasets
from experiments import validate_checkpoint
from models.simple_cnn import SimpleCNN

NAMES = {"Cr": "Crazing", "In": "Inclusion", "Pa": "Patches",
         "PS": "Pitted surface", "RS": "Rolled-in scale", "Sc": "Scratches"}
CLASSES = list(NAMES)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir
    figures = ROOT / "docs/images"
    results = ROOT / "results/baseline_run_001"
    figures.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / "data/splits/NEU-CLS_deduplicated_v1.json"
    data_dir, manifest = load_split_manifest(args.data_dir, manifest_path)
    metrics = json.loads((run / "test_metrics.json").read_text(encoding="utf-8"))
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    with (run / "test_predictions.csv").open(encoding="utf-8-sig", newline="") as stream:
        predictions = list(csv.DictReader(stream))
    with (run / "history.csv").open(encoding="utf-8", newline="") as stream:
        history = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]

    # Independently check all 270 saved test predictions using the actual checkpoint.
    torch.set_num_threads(min(8, torch.get_num_threads()))
    checkpoint = torch.load(run / "best_model.pth", map_location="cpu", weights_only=True)
    validate_checkpoint(checkpoint, "simple_cnn", manifest["split_id"])
    assert config["split_id"] == metrics["split_id"] == manifest["split_id"]
    model = SimpleCNN().eval()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    _, _, test = create_datasets(data_dir, manifest_path)
    assert [row["file"] for row in predictions] == [path.name for path in test.image_paths]
    matrix = np.zeros((6, 6), dtype=int)
    offset = 0
    max_score_difference = 0.0
    with torch.inference_mode():
        for images, labels in torch.utils.data.DataLoader(test, batch_size=16):
            scores = model(images).softmax(1)
            for label, score in zip(labels.tolist(), scores):
                guess = int(score.argmax())
                row = predictions[offset]
                assert row["true"] == IDX_TO_CLASS[label]
                assert row["predicted"] == IDX_TO_CLASS[guess]
                assert (row["correct"] == "True") == (label == guess)
                # CPU and GPU kernels can produce slightly different scores.
                # Require identical class predictions; figures retain recorded scores.
                assert 0 <= float(row["confidence"]) <= 1
                max_score_difference = max(max_score_difference, abs(float(row["confidence"]) - float(score[guess])))
                matrix[label, guess] += 1
                offset += 1
    assert offset == metrics["total"] == len(test)
    assert int(matrix.trace()) == metrics["correct"]
    assert np.array_equal(matrix, np.asarray(metrics["confusion_matrix"]))
    assert abs(matrix.trace() / offset - metrics["accuracy"]) < 1e-12
    assert checkpoint["best_epoch"] == metrics["best_epoch"]
    print(f"Verified all {offset} test predictions: {matrix.trace()}/{offset} correct.", flush=True)
    print(f"Largest score difference from recorded run: {max_score_difference:.6g}", flush=True)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})

    fig, axes = plt.subplots(2, 3, figsize=(10, 7.5))
    for axis, code in zip(axes.flat, CLASSES):
        path = data_dir / f"{code}_1.bmp"
        with Image.open(path) as image:
            assert image.size == (200, 200)
            axis.imshow(image.convert("L"), cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        axis.set_title(f"{NAMES[code]} ({code})\n{path.name}", fontsize=11)
        axis.axis("off")
    fig.suptitle("Original NEU-CLS images | 200 x 200 grayscale", fontsize=17, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=4)
    fig.savefig(figures / "original_samples.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    epochs = [int(row["epoch"]) for row in history]
    for prefix, label, color in (("train", "Training", "#2563eb"), ("val", "Validation", "#d97706")):
        axes[0].plot(epochs, [row[prefix + "_accuracy"] for row in history], "o-", label=label, color=color)
        axes[1].plot(epochs, [row[prefix + "_loss"] for row in history], "o-", label=label, color=color)
    axes[0].set(title="Classification accuracy", ylabel="Accuracy", ylim=(0, 1))
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].set(title="Cross-entropy loss", ylabel="Loss")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.set_xticks(epochs)
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    fig.suptitle("Baseline CNN | 10 epochs, trained from random weights", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(figures / "training_curves.png", dpi=170)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.7, 6.5))
    grid = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=45)
    fig.colorbar(grid, ax=axis, label="Number of test images", shrink=0.8)
    for row in range(6):
        for column in range(6):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center",
                      color="white" if matrix[row, column] > 22 else "#172554", fontsize=13)
    axis.set(xticks=range(6), yticks=range(6), xticklabels=CLASSES, yticklabels=CLASSES,
             xlabel="Predicted class", ylabel="True class")
    axis.set_title(f"Test confusion matrix\n{metrics['correct']}/{metrics['total']} correct | {metrics['accuracy']:.2%} accuracy", pad=15, fontsize=15)
    fig.tight_layout()
    fig.savefig(figures / "confusion_matrix.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 6, figsize=(16, 6))
    for column, code in enumerate(CLASSES):
        for row_index, correct in enumerate((True, False)):
            axis = axes[row_index, column]
            examples = [row for row in predictions if row["true"] == code and (row["correct"] == "True") == correct]
            axis.axis("off")
            if not examples:
                axis.text(0.5, 0.5, f"{code}: no test errors\n45 / 45 correct", ha="center", va="center", transform=axis.transAxes, color="#15803d")
                continue
            example = examples[0]
            with Image.open(data_dir / example["file"]) as image:
                axis.imshow(image.convert("L"), cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            axis.set_title(f"True: {example['true']} | Pred: {example['predicted']}\nScore: {float(example['confidence']):.1%}",
                           fontsize=10, color="#15803d" if correct else "#b91c1c")
            axis.text(0.5, -0.04, example["file"], ha="center", va="top", transform=axis.transAxes, fontsize=9)
    fig.suptitle("Actual test predictions | correct examples (top), errors (bottom)", fontsize=16, fontweight="bold")
    fig.text(0.5, 0.015, "First matching test image per true class, in filename order. Score = maximum softmax probability.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93), h_pad=3)
    fig.savefig(figures / "prediction_examples.png", dpi=170)
    plt.close(fig)

    for filename in ("config.json", "history.csv", "test_metrics.json", "test_predictions.csv", "training_summary.json", "data_audit.json"):
        shutil.copy2(run / filename, results / filename)
    print("Saved four README figures and six original experiment records.")


if __name__ == "__main__":
    main()
