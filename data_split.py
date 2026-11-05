"""A persisted, pixel-deduplicated train/validation/test split."""
import hashlib
import json
from collections import Counter
from pathlib import Path

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_DIR / "data/NEU-CLS"
CLASS_TO_IDX = {"Cr": 0, "In": 1, "Pa": 2, "PS": 3, "RS": 4, "Sc": 5}
IDX_TO_CLASS = {value: key for key, value in CLASS_TO_IDX.items()}
SPLIT_NAMES = ("train", "validation", "test")


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def image_digest(path):
    # Hash what the model reads; differently encoded copies are duplicates too.
    with Image.open(path) as source:
        image = source.convert("L")
        return hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()


def scan_images(data_dir):
    records = []
    for path in sorted(data_dir.glob("*.bmp"), key=lambda p: p.name):
        class_name = path.stem.split("_")[0]
        if class_name not in CLASS_TO_IDX:
            raise ValueError(f"Unknown class: {path.name}")
        records.append({"path": path.name, "label": CLASS_TO_IDX[class_name],
                        "pixel_sha256": image_digest(path)})
    if not records:
        raise ValueError(f"No BMP images found in {data_dir}")
    return records


def deduplicate(records):
    unique, excluded, seen = [], [], {}
    for record in records:
        previous = seen.get(record["pixel_sha256"])
        if previous is None:
            seen[record["pixel_sha256"]] = record
            unique.append(record)
        else:
            if record["label"] != previous["label"]:
                raise ValueError(f"Identical images have conflicting labels: {previous['path']}, {record['path']}")
            excluded.append({"path": record["path"], "kept": previous["path"],
                             "pixel_sha256": record["pixel_sha256"]})
    return unique, excluded


def validate_manifest(manifest, records):
    if manifest.get("version") != 1 or manifest.get("dataset_sha256") != digest_json(records):
        raise ValueError("Dataset changed since the saved split was created. Use a new manifest and retrain; do not reuse old checkpoints.")
    payload = {k: v for k, v in manifest.items() if k != "split_id"}
    if manifest.get("split_id") != digest_json(payload):
        raise ValueError("Split manifest was edited or corrupted.")
    unique, excluded = deduplicate(records)
    if manifest["excluded_duplicates"] != excluded:
        raise ValueError("Manifest duplicate list does not match image contents.")
    assigned = [r for name in SPLIT_NAMES for r in manifest["splits"][name]]
    if sorted(assigned, key=lambda r: r["path"]) != unique:
        raise ValueError("Every unique image must appear exactly once in the split.")
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        left, right = manifest["splits"][a], manifest["splits"][b]
        if {r["path"] for r in left} & {r["path"] for r in right}:
            raise ValueError(f"Path leakage: {a}/{b}")
        if {r["pixel_sha256"] for r in left} & {r["pixel_sha256"] for r in right}:
            raise ValueError(f"Pixel-content leakage: {a}/{b}")
    if any(set(r["label"] for r in manifest["splits"][name]) != set(IDX_TO_CLASS) for name in SPLIT_NAMES):
        raise ValueError("Every split must contain all six classes.")


def load_split_manifest(data_dir=None, manifest_path=None):
    data_dir = Path(data_dir or DEFAULT_DATA_DIR).resolve()
    manifest_path = Path(manifest_path) if manifest_path else data_dir.parent / "splits" / f"{data_dir.name}_deduplicated_v1.json"
    records = scan_images(data_dir)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        unique, excluded = deduplicate(records)
        train, temporary = train_test_split(unique, test_size=0.30, random_state=42,
                                            stratify=[r["label"] for r in unique])
        validation, test = train_test_split(temporary, test_size=0.50, random_state=42,
                                            stratify=[r["label"] for r in temporary])
        manifest = {"version": 1, "seed": 42, "policy": "Keep one image per grayscale pixel SHA-256; stratify 70/15/15.",
                    "dataset_sha256": digest_json(records), "source_count": len(records),
                    "excluded_duplicates": excluded,
                    "splits": {name: sorted(items, key=lambda r: r["path"])
                               for name, items in zip(SPLIT_NAMES, (train, validation, test))}}
        manifest["split_id"] = digest_json(manifest)
        validate_manifest(manifest, records)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2)
    validate_manifest(manifest, records)
    return data_dir, manifest


class NEUDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, split_id=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.split_id = split_id

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        with Image.open(self.image_paths[index]) as source:
            image = source.convert("L")
        if self.transform:
            image = self.transform(image)
        return image, self.labels[index]


def make_datasets(train_transform, eval_transform, data_dir=None, manifest_path=None):
    data_dir, manifest = load_split_manifest(data_dir, manifest_path)
    return tuple(NEUDataset([data_dir / r["path"] for r in manifest["splits"][name]],
                            [r["label"] for r in manifest["splits"][name]],
                            train_transform if name == "train" else eval_transform,
                            split_id=manifest["split_id"]) for name in SPLIT_NAMES)


def make_dataloaders(datasets, batch_size=16):
    return tuple(DataLoader(ds, batch_size=batch_size, shuffle=(i == 0))
                 for i, ds in enumerate(datasets))


def split_summary(manifest):
    overlaps = {}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlaps[a + "/" + b] = len({r["pixel_sha256"] for r in manifest["splits"][a]} &
                                     {r["pixel_sha256"] for r in manifest["splits"][b]})
    return {"split_id": manifest["split_id"], "source_count": manifest["source_count"],
            "unique_count": sum(map(len, manifest["splits"].values())),
            "excluded_duplicates": manifest["excluded_duplicates"],
            "split_counts": {name: len(items) for name, items in manifest["splits"].items()},
            "class_counts": {name: dict(Counter(IDX_TO_CLASS[r["label"]] for r in items))
                             for name, items in manifest["splits"].items()},
            "cross_split_pixel_overlap": overlaps}


if __name__ == "__main__":
    _, manifest = load_split_manifest()
    print(json.dumps(split_summary(manifest), ensure_ascii=False, indent=2))
