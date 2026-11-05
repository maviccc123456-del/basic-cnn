"""Baseline images using the shared, deduplicated split."""
from torchvision import transforms
from data_split import CLASS_TO_IDX, IDX_TO_CLASS, NEUDataset, make_datasets, make_dataloaders


def create_datasets(data_dir=None, manifest_path=None):
    transform = transforms.ToTensor()
    return make_datasets(transform, transform, data_dir, manifest_path)


def create_dataloaders(batch_size=16):
    return make_dataloaders(create_datasets(), batch_size)


if __name__ == "__main__":
    for name, loader in zip(("Train", "Validation", "Test"), create_dataloaders()):
        print(name, len(loader.dataset))
