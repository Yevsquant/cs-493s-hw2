import random
from sympy import mod_inverse
from pathlib import Path

ps = [97, 113]

def generate_data(p, num_samples):
    data = []
    ops = ['/']

    while len(data) < num_samples:
        # randomly choose a, b, and the operator
        a = random.randint(0, p)
        b = random.randint(1, p)
        op = random.choice(ops)

        # find c
        if op == '+':
            c = (a + b) % p
        elif op == '-':
            c = (a - b) % p
        elif op == '/':
            try:
                b_inv = mod_inverse(b, p)
                c = (a * b_inv) % p
            except ValueError:
                continue

        data.append(f"{a} {op} {b} = {c}")

    return data

def split_and_save(data, path_prefix, fname):
    """
    |--train--|--val--|--test--|
    train is 40%, val is 40%, and test is 20%
    """
    random.shuffle(data)
    n = len(data)
    train_end = int(n * 0.4)
    val_end = int(n * 0.8)

    path_prefix.mkdir(parents=True, exist_ok=True)

    with open(path_prefix / f"traind_{fname}.txt", "w") as f:
        f.write("\n".join(data[:train_end]))

    with open(path_prefix / f"vald_{fname}.txt", "w") as f:
        f.write("\n".join(data[train_end:val_end]))

    with open(path_prefix / f"testd_{fname}.txt", "w") as f:
        f.write("\n".join(data[val_end:]))

def main():
    data = []
    total_samples_per_p = 5000  # There are in total 113 * 113 * 3 + 97 * 97 * 3 = 66534 samples
    for p in ps:
        data.extend(generate_data(p, total_samples_per_p))
        split_and_save(data, Path(f"./AlgorithmicTasks"), f"p{p}")

    split_and_save(data, Path(f"./AlgorithmicTasks"), "all")

if __name__ == "__main__":
    main()
