## Part 1
![d798bfb2d7cd0b08ad887dc5b9d99cb](https://github.com/user-attachments/assets/ada5a754-f6e6-49ec-8c19-fefa2747c77b)
<img width="969" alt="199e869f4678f994b9b9fecfe46d78d" src="https://github.com/user-attachments/assets/c9294e5b-cbf8-423d-8410-b20b5455383d" />

## Part 2
### 2.1 Data Generation
We have 5000 samples for p=97 and p=113, respectively. For each p, the size of the train samples is 2000, the size of the validation samples is 2000, and the size of the test is 1000. 

```
    ops = ['+', '-', '/']
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

    data = f"{a}{op}{b}={c}"
```
The above code is how we generate a single sample. Then we repeat this process 5000 times for each `p`.
After that, we split the data via the following code (5000*0.4=2000) and save the data in `txt` file: \
all samples = |--train--|--val--|--test--| \
train is 40%, val is 40%, and test is 20%
```
random.shuffle(data)
n = len(data)
train_end = int(n * 0.4)
val_end = int(n * 0.8)

path_prefix.mkdir(parents=True, exist_ok=True)

with open(path_prefix / f"train_{fname}.txt", "w") as f:
    f.write("\n".join(data[:train_end]))

with open(path_prefix / f"val_{fname}.txt", "w") as f:
    f.write("\n".join(data[train_end:val_end]))

with open(path_prefix / f"test_{fname}.txt", "w") as f:
    f.write("\n".join(data[val_end:]))
```

### 2.2 Warmup - Addition and Subtraction Experiments


### 2.3 Grokking


### 2.4 Ablations/Analysis


