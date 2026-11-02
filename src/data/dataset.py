from datasets import load_dataset

CACHE_DIR = "data/hf_cache"


def load_cub():
    """Load CUB-200-2011 train/test splits from Hugging Face."""
    train = load_dataset("Multimodal-Fatima/CUB_train", split="train", cache_dir=CACHE_DIR)
    test = load_dataset("Multimodal-Fatima/CUB_test", split="test", cache_dir=CACHE_DIR)
    return train, test


def get_class_names(ds):
    """Return the list of 200 species names (works for ClassLabel or plain strings)."""
    feat = ds.features["label"]
    if hasattr(feat, "names"):
        return feat.names
    return sorted(set(ds["label"]))


def split_captions(description):
    """The description column holds 10 captions separated by newlines."""
    return [c.strip() for c in description.split("\n") if c.strip()]

from torch.utils.data import Dataset


class CubClassificationDataset(Dataset):
    def __init__(self, hf_ds, indices, class_names, transform=None):
        self.ds = hf_ds
        self.indices = list(indices)
        self.name_to_id = {n: i for i, n in enumerate(class_names)}
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        item = self.ds[self.indices[i]]
        img = item["image"].convert("RGB")  # fixes the 8 grayscale images
        label = item["label"]
        if isinstance(label, str):
            label = self.name_to_id[label]
        if self.transform:
            img = self.transform(img)
        return img, label