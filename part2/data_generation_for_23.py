import random
from sympy import mod_inverse
from pathlib import Path

ps = [97]

def generate_data(p, num_samples):
    data = []
    op = '/'

    for a in range(p+1):
        for b in range(1,p+1):
            try:
                b_inv = mod_inverse(b, p)
                c = (a * b_inv) % p
            except ValueError:
                continue
            data.append(f"{a}{op}{b}={c}")

    while len(data) < num_samples:
        # randomly choose a, b, and the operator
        a = random.randint(0, p)
        b = random.randint(1, p)

        # find c 
        try:
            b_inv = mod_inverse(b, p)
            c = (a * b_inv) % p
        except ValueError:
            continue

        data.append(f"{a}{op}{b}={c}")

    return data

def split_and_save(data, path_prefix, fname):
    """
    |--train--|--val--|--test--|
    train is 40%, val is 40%, and test is 20%
    """
    random.shuffle(data)
    n = len(data)
    train_end = int(n * 0.05)
    val_end = int(n * 0.1)
    test_end = int(n * 0.15)

    path_prefix.mkdir(parents=True, exist_ok=True)

    with open(path_prefix / f"train_division_{fname}_all500.txt", "w") as f:
        f.write("\n".join(data[:train_end]))

    with open(path_prefix / f"val_division_{fname}_all500.txt", "w") as f:
        f.write("\n".join(data[train_end:val_end]))

    with open(path_prefix / f"test_division_{fname}_all500.txt", "w") as f:
        f.write("\n".join(data[val_end:test_end]))

def main():
    data = []
    total_samples_per_p = 10000  # There are in total 113 * 113 * 3 + 97 * 97 * 3 = 66534 samples
    for p in ps:
        data.extend(generate_data(p, total_samples_per_p))
        split_and_save(data, Path(f"./AlgorithmicTasks"), f"{total_samples_per_p}")

    # split_and_save(data, Path(f"./AlgorithmicTasks"), "mix")

if __name__ == "__main__":
    main()

"""
import numpy as np

# Assume halton_sequence() is defined from earlier
halton_points = halton_sequence(size=10, dim=2)

low, high = 10, 50
scaled_ints = [ [int(low + x*(high - low)) for x in point] for point in halton_points ]

for i, point in enumerate(scaled_ints):
    print(f"Sample {i}: {point}")
"""