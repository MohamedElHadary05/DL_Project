# train / val / test DataLoaders for classification
import torch
from torch.utils.data import DataLoader

from src.data.dataset import CubClassificationDataset, get_class_names, load_cub
from src.data.preprocessing import SEED, clean_test_indices, stratified_split
from src.data.transforms import get_eval_transform, get_train_transform


def build_classification_loaders(image_size=128, batch_size=64, num_workers=2,
                                 val_size=0.2, seed=SEED):
    """Return (train_loader, val_loader, test_loader, class_names).

    Val is a stratified slice of the official train split (no augmentation);
    test is the official test split minus the leaked near-duplicate.
    """
    train_hf, test_hf = load_cub()
    class_names = get_class_names(train_hf)
    train_idx, val_idx = stratified_split(train_hf["label"], val_size, seed)

    datasets = {
        "train": CubClassificationDataset(train_hf, train_idx, class_names,
                                          get_train_transform(image_size)),
        "val": CubClassificationDataset(train_hf, val_idx, class_names,
                                        get_eval_transform(image_size)),
        "test": CubClassificationDataset(test_hf, clean_test_indices(len(test_hf)),
                                         class_names, get_eval_transform(image_size)),
    }
    loaders = {
        name: DataLoader(ds, batch_size=batch_size, shuffle=(name == "train"),
                         num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                         persistent_workers=num_workers > 0)
        for name, ds in datasets.items()
    }
    return loaders["train"], loaders["val"], loaders["test"], class_names
