## Part 1
The log is [here](/handin/part1_log.txt). Model checkpoint is [here](/part1). `train.py` is [here](/part1). `inference.py` is [here](/part1). \
Go to part1, \
For training,
```
python train.py textfile.txt \
    --n_layer 1 \
    --n_head 2 \
    --n_embd 32 \
    --block_size 16 \
    --batch_size 1 \
    --out_dir ckpt
```

For inference, `python inference.py ckpt/model.pt --prompt "I" --steps 22` \
To overfit "I love machine learning", we set the configuaration of GPT smaller than its default to avoid the model taking this as an complex task. We were doing character-level tokenization and wrote corespponding char dataset. We used AdamW as our optimizer and F.cross_entropy as the criterion. We trained this single text over 5000 epochs and make sure it was overfitting.

## Part 2
### 2.1 Data Generation
We have 5000 samples for p=97 and p=113, respectively. For each p, the size of the train samples is 2000, the size of the validation samples is 2000, and the size of the test is 1000. 

```
    ops = ['+', '-']
    a = random.randint(0, p)
    b = random.randint(1, p)
    op = random.choice(ops)

    # find c
    if op == '+':
        c = (a + b) % p
    else:
        c = (a - b) % p

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
seed=493_train_loss
![seed=493_train_loss](493_all_train_loss.png) \
seed=493_test_loss
![seed=493_val_loss](493_all_val_loss.png) \
seed=493_train_acc
![seed=493_train_acc](493_all_train_acc.png) \
seed=493_test_acc
![seed=493_val_acc](493_all_val_acc.png) \
seed=599_train_loss
![seed=599_train_loss](599_all_train_loss.png) \
seed=599_test_loss
![seed=599_val_loss](599_all_val_loss.png) \
seed=599_train_acc
![seed=599_train_acc](599_all_train_acc.png) \
seed=599_test_acc
![seed=599_val_acc](599_all_val_acc.png) \
seed=42_train_loss
![seed=42_train_loss](42_all_train_loss.png) \
seed=42_test_loss
![seed=42_val_loss](42_all_val_loss.png) \
seed=42_train_acc
![seed=42_train_acc](42_all_train_acc.png) \
seed=42_test_acc
![seed=42_val_acc](42_all_val_acc.png) \
Final loss is 0.0, and accuracy is 1.0 across three seeds. \
[Checkpoint](/models/mod_checkpoints_warmup/model) \
(There might be error in formating the metrics computing the accuracies and losses; also, train acc is computed through batches, while validation acc is computed across the entire validation dataset, we replace it with other code in 2.3)

### 2.3 Grokking
![loss](lossd23.png) \
![acc](accd23.png) \
The checkpoint is [here](/part2/ckpt_task22/) of seed=42. \
For training: \
```
python3 train23.py . \
    --prime 97 \
    --operators "/" \
    --n_layer 2 \
    --max_iters 10000 \
    --eval_interval 100 \
    --batch_size 32 \
    --device cpu
```

### 2.4 Ablations/Analysis
Increase the size of the dataset helps gorkking on division task faster and reliable in validation, not training. As shown,
![dataset](largedataset.png) \
Under the condition of small samples, the model quickly remembers the training set (Acc is about 1.0), but due to insufficient patterns, extrapolation is impossible, and the validation set remains near random guessing (about 0.49). This is precisely what is called the silent period in grokking's literature

In the large sample setting, although the memory stage slows down, the verification accuracy significantly improves with training. The gap of the training-verification curve Narrows from approximately 0.5 initially to approximately 0.38, indicating that the model learns the generalizable internal representation earlier.

Therefore, in the arithmetic grokking task, the number of training samples is the primary factor determining the generalization speed and final performance. If hardware and time permit, prioritizing the increase of data volume is more effective in eliminating the grokking phenomenon than adjusting the learning rate or regularization in isolation.

