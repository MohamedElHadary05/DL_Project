# sanity check that every teammate has the same env, data and split
import hashlib
import sys

import torch

from src.data.dataset import get_class_names, load_cub
from src.data.preprocessing import clean_test_indices, stratified_split


def main():
    train, test = load_cub()
    names = get_class_names(train)
    train_idx, val_idx = stratified_split(train["label"])
    test_idx = clean_test_indices(len(test))
    fingerprint = hashlib.md5(",".join(map(str, sorted(val_idx))).encode()).hexdigest()

    print(f"python      {sys.version.split()[0]}")
    print(f"torch       {torch.__version__}")
    print(f"cuda        {torch.cuda.is_available()}")
    print(f"classes     {len(names)}")
    print(f"train/val   {len(train_idx)} {len(val_idx)}")
    print(f"test        {len(test_idx)}")
    print(f"val split fingerprint {fingerprint[:12]}")

    assert len(names) == 200
    assert (len(train_idx), len(val_idx), len(test_idx)) == (4795, 1199, 5793)
    print("ENV OK")


if __name__ == "__main__":
    main()
