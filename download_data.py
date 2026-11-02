from src.data.dataset import load_cub, get_class_names


def main():
    print("Downloading CUB-200-2011 (about 1.1 GB on first run)...")
    train, test = load_cub()
    names = get_class_names(train)
    print(f"train={len(train)} test={len(test)} classes={len(names)}")

    assert len(train) == 5994, "unexpected train size"
    assert len(test) == 5794, "unexpected test size"
    assert len(names) == 200, "unexpected class count"
    print("Dataset OK. Cached in data/hf_cache")


if __name__ == "__main__":
    main()