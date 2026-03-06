import numpy as np
import os

def create_splits(df, split_dir, seed=42,x=200):
    np.random.seed(seed) 
    n = len(df)
    indices = np.arange(n)
    np.random.shuffle(indices)
    if x is not None and x < n:
        indices = indices[:x]
        n = x
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:n]
    test_01_size = int(0.1 * len(test_idx))
    test_01_idx = test_idx[:test_01_size]
    os.makedirs(split_dir, exist_ok=True)
    def write_indices(name, idx):
        with open(os.path.join(split_dir, f"{name}.index"), "w") as f:
            for i in idx:
                f.write(f"{int(i)}\n")
    write_indices("train", train_idx)
    write_indices("val", val_idx)
    write_indices("test", test_idx)
    write_indices("test-0.1", test_01_idx)