# caption tokenization + vocabulary (built on the train split only)
import json
import re
from collections import Counter

from src.data.dataset import split_captions
from src.data.preprocessing import clean_captions

PAD, START, END, UNK = "<pad>", "<start>", "<end>", "<unk>"
SPECIALS = [PAD, START, END, UNK]


def tokenize(caption):
    """Words only: punctuation carries no meaning for the decoder."""
    return re.findall(r"[a-z0-9']+", caption)


def captions_for(description):
    """Raw description field -> list of cleaned captions."""
    return clean_captions(split_captions(description))


class Vocabulary:
    def __init__(self, words):
        self.itos = SPECIALS + [w for w in words if w not in SPECIALS]
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    @classmethod
    def build(cls, captions, min_freq=5):
        """Words seen fewer than min_freq times (mostly typos, see
        outputs/eda/typo_candidates_train.csv) are mapped to <unk>."""
        counts = Counter(w for c in captions for w in tokenize(c))
        words = sorted(w for w, n in counts.items() if n >= min_freq)
        return cls(words)

    def encode(self, caption, max_len=None):
        ids = [self.stoi.get(w, self.stoi[UNK]) for w in tokenize(caption)]
        if max_len:
            ids = ids[:max_len - 2]
        return [self.stoi[START]] + ids + [self.stoi[END]]

    def decode(self, ids):
        words = []
        for i in ids:
            w = self.itos[i]
            if w == END:
                break
            if w not in (PAD, START):
                words.append(w)
        return " ".join(words)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.itos, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(json.load(f))


def main():
    from pathlib import Path

    from src.data.dataset import load_cub
    from src.data.preprocessing import stratified_split

    train, _ = load_cub()
    train_idx, _ = stratified_split(train["label"])
    descriptions = train.select(train_idx)["description"]

    raw = [c for d in descriptions for c in split_captions(d)]
    kept = [c for d in descriptions for c in captions_for(d)]
    vocab = Vocabulary.build(kept)

    tokens = [w for c in kept for w in tokenize(c)]
    unk = sum(1 for w in tokens if w not in vocab.stoi)
    lengths = sorted(len(tokenize(c)) for c in kept)
    print(f"captions raw={len(raw)} kept={len(kept)} dropped={len(raw) - len(kept)}")
    print(f"vocab size={len(vocab)} (min_freq=5), <unk> rate={unk / len(tokens):.2%}")
    print(f"length 95th percentile={lengths[int(0.95 * len(lengths))]} words")

    out = Path("outputs")
    out.mkdir(exist_ok=True)
    vocab.save(out / "vocab.json")
    print(f"saved {out / 'vocab.json'}")


if __name__ == "__main__":
    main()
