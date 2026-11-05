"""Training and evaluation bound to the exact deduplicated split used by a run."""
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from data_split import PROJECT_DIR, IDX_TO_CLASS, load_split_manifest, split_summary


MODEL_NAMES = ("simple_cnn",)
OUTPUT_DIR = PROJECT_DIR / "outputs/leak_free"


def datasets_for(model_name):
    if model_name != "simple_cnn":
        raise ValueError(f"Unknown model: {model_name}")
    from dataset import create_datasets
    return create_datasets()


def create_model(model_name, pretrained=False):
    if model_name != "simple_cnn":
        raise ValueError(f"Unknown model: {model_name}")
    if pretrained:
        raise ValueError("The baseline CNN does not use pretrained weights.")
    from models.simple_cnn import SimpleCNN
    return SimpleCNN(num_classes=6)


def validate_checkpoint(checkpoint, model_name, split_id):
    if checkpoint.get("format_version") != 1 or "model_state_dict" not in checkpoint:
        raise ValueError("Legacy/untracked weights cannot be evaluated on the new split. Retrain first.")
    if checkpoint.get("model_name") != model_name:
        raise ValueError("Checkpoint model does not match the requested model.")
    if checkpoint.get("split_id") != split_id:
        raise ValueError("Checkpoint was trained on another split. Retrain on the current manifest.")


def load_model_for_evaluation(model_name, split_id, device, run_dir=None):
    if run_dir is None:
        marker = OUTPUT_DIR / model_name / "latest.json"
        if not marker.exists():
            raise FileNotFoundError(f"No completed training for {model_name} on the deduplicated split. Run its training script first.")
        run_dir = marker.parent / json.loads(marker.read_text(encoding="utf-8"))["run"]
    run_dir = Path(run_dir)
    checkpoint = torch.load(run_dir / "best_model.pth", map_location="cpu", weights_only=True)
    validate_checkpoint(checkpoint, model_name, split_id)
    # Evaluation restores the weights trained for this exact data split.
    model = create_model(model_name, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval(), checkpoint, run_dir


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def new_run_dir(model_name):
    parent = OUTPUT_DIR / model_name
    parent.mkdir(parents=True, exist_ok=True)
    index = 1
    while True:
        run_dir = parent / f"run_{index:03d}"
        try:
            run_dir.mkdir()
            return run_dir
        except FileExistsError:
            index += 1


def run_training(model_name, num_epochs=10, batch_size=16, seed=42):
    if num_epochs < 1:
        raise ValueError("num_epochs must be positive")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, manifest = load_split_manifest()
    train_ds, val_ds, _ = datasets_for(model_name)
    assert train_ds.split_id == val_ds.split_id == manifest["split_id"]
    generator = torch.Generator().manual_seed(seed)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    model = create_model(model_name).to(device)
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    run_dir = new_run_dir(model_name)
    config = {"model_name": model_name, "num_epochs": num_epochs, "batch_size": batch_size,
              "learning_rate": learning_rate, "seed": seed, "split_id": manifest["split_id"],
              "initialization": "random weights",
              "torch_version": str(torch.__version__), "device": str(device)}
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "split_manifest.json", manifest)
    write_json(run_dir / "data_audit.json", split_summary(manifest))
    best_accuracy, best_epoch = -1.0, 0
    print(f"Training {model_name} from fresh initialization on {device}; output: {run_dir}", flush=True)
    with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        history = csv.DictWriter(stream, fieldnames=["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"])
        history.writeheader()
        for epoch in range(1, num_epochs + 1):
            model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0
            for batch, (images, labels) in enumerate(train_loader, start=1):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * labels.size(0)
                train_correct += (logits.argmax(1) == labels).sum().item()
                train_total += labels.size(0)
                if batch % 20 == 0:
                    print(f"Epoch {epoch}/{num_epochs}: trained {train_total}/{len(train_ds)} images", flush=True)
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.inference_mode():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    logits = model(images)
                    val_loss += criterion(logits, labels).item() * labels.size(0)
                    val_correct += (logits.argmax(1) == labels).sum().item()
                    val_total += labels.size(0)
            row = {"epoch": epoch, "train_loss": train_loss / train_total,
                   "train_accuracy": train_correct / train_total,
                   "val_loss": val_loss / val_total, "val_accuracy": val_correct / val_total}
            history.writerow(row)
            stream.flush()
            print(f"Epoch {epoch}/{num_epochs}: train {row['train_accuracy']:.2%}, validation {row['val_accuracy']:.2%}", flush=True)
            if row["val_accuracy"] > best_accuracy:
                best_accuracy, best_epoch = row["val_accuracy"], epoch
                torch.save({"format_version": 1, "model_name": model_name,
                            "model_state_dict": model.state_dict(), "split_id": manifest["split_id"],
                            "best_epoch": best_epoch, "best_val_accuracy": best_accuracy,
                            "config": config}, run_dir / "best_model.pth")
    write_json(run_dir / "training_summary.json", {"best_epoch": best_epoch, "best_val_accuracy": best_accuracy,
                                                  "split_id": manifest["split_id"], "status": "completed"})
    write_json(run_dir.parent / "latest.json", {"run": run_dir.name, "split_id": manifest["split_id"]})
    print("Training completed; test set has not been used for model selection.", flush=True)
    return run_dir


def run_evaluation(model_name, run_dir=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    torch.set_num_threads(min(8, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, manifest = load_split_manifest()
    _, _, dataset = datasets_for(model_name)
    assert dataset.split_id == manifest["split_id"]
    model, checkpoint, run_dir = load_model_for_evaluation(model_name, dataset.split_id, device, run_dir)
    truth, predictions, rows = [], [], []
    with torch.inference_mode():
        for images, labels in torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False):
            probabilities = model(images.to(device)).softmax(1).cpu()
            guesses = probabilities.argmax(1).tolist()
            for j, (label, guess) in enumerate(zip(labels.tolist(), guesses)):
                path = dataset.image_paths[len(rows)]
                rows.append({"file": path.name, "true": IDX_TO_CLASS[label], "predicted": IDX_TO_CLASS[guess],
                             "confidence": float(probabilities[j, guess]), "correct": label == guess})
            truth.extend(labels.tolist())
            predictions.extend(guesses)
    class_names = [IDX_TO_CLASS[i] for i in range(6)]
    matrix = confusion_matrix(truth, predictions, labels=list(range(6)))
    correct = int((np.asarray(truth) == predictions).sum())
    metrics = {"model_name": model_name, "split_id": dataset.split_id, "total": len(truth), "correct": correct,
               "accuracy": correct / len(truth), "best_epoch": checkpoint["best_epoch"],
               "best_val_accuracy": checkpoint["best_val_accuracy"], "confusion_matrix": matrix.tolist(),
               "classification_report": classification_report(truth, predictions, labels=list(range(6)),
                                                               target_names=class_names, output_dict=True, zero_division=0)}
    write_json(run_dir / "test_metrics.json", metrics)
    with (run_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    display = ConfusionMatrixDisplay(matrix, display_labels=class_names)
    display.plot(cmap="Blues", values_format="d")
    plt.title(f"{model_name} - deduplicated split")
    plt.tight_layout()
    plt.savefig(run_dir / "confusion_matrix.png", dpi=200)
    plt.close()
    print(f"{model_name} test: {correct}/{len(truth)} = {metrics['accuracy']:.2%}; saved to {run_dir}", flush=True)
    return metrics
