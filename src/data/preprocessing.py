# caption splitting / cleaning, train-val split
import re

from sklearn.model_selection import train_test_split

SEED = 42

# Test image 661 is a near-duplicate of train image 774 (dHash match, visually confirmed).
LEAKED_TEST_IDX = {661}


def stratified_split(labels, val_size=0.2, seed=SEED):
    """Return (train_idx, val_idx) keeping each class's proportion."""
    idx = list(range(len(labels)))
    train_idx, val_idx = train_test_split(
        idx, test_size=val_size, stratify=labels, random_state=seed
    )
    return train_idx, val_idx


def clean_test_indices(n_test):
    """Test-set indices with the leaked near-duplicate removed."""
    return [i for i in range(n_test) if i not in LEAKED_TEST_IDX]


def clean_caption(text):
    """Non-ASCII chars become spaces (not deleted), then lowercase and tidy."""
    text = "".join(" " if ord(ch) > 127 else ch for ch in text)
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s,.']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_captions(captions, min_words=3, max_words=40):
    """Clean, drop too-short / too-long captions."""
    out = []
    for c in captions:
        c = clean_caption(c)
        if min_words <= len(c.split()) <= max_words:
            out.append(c)
    return out
