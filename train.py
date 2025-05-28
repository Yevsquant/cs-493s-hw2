import random
import math
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from .model import GPT, GPTConfig

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = None

# Seed all randomness for reproducibility
seed = 42
# TODO the function is emm ...
def set_seed():
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.random.manual_seed(seed)
set_seed()

# Load custom models
def get_llm(model_name, cache_dir="models", device="cpu"):
    model = AutoModelForCausalLM.from_pretrained(
        cache_dir + "/" + model_name,
        # torch_dtype=torch.float16
    )
    model.to(device)
    return model

# data for training
# temprary data for testing the infrastructure of train.py
texts = ["I love machine learning",
         "There is no apple",
         "This is not a serious program"]

def build_dataset():
    encodings = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    exponent = round(math.log2(len(encodings["input_ids"][0])))
    # find a proper block size which should be less than model.config.block_size
    block_size = 2 ** exponent
    if model is not None and block_size > model.config.block_size:
        block_size = model.config.block_size
    if model is None and block_size > 16: # hardcode max block size
        block_size = 16

    x, y, attn_mask = [], [], []
    # TODO: might vectorize part of the process to speed up
    for i in range(len(texts)):
        token = encodings["input_ids"][i]  # shape: (T,)
    
        for j in range(0, len(token) - block_size + 1, block_size):
            if token[j] == tokenizer.pad_token_id:
                continue
            x.append(token[j:j+block_size])
            y.append(token[j+1:j+1+block_size])
            attn_mask.append(encodings['attention_mask'][i][j:j+block_size])
        
        # short sequence with padding (T < block_size)
        leftover = len(token) % block_size
        if leftover > 1:
            idx_for_short_seq = len(token) - leftover
            diff = block_size - leftover
            x_ = torch.cat([token[idx_for_short_seq:idx_for_short_seq+block_size], torch.tensor([tokenizer.pad_token_id] * diff)], dim=0)
            y_ = torch.cat([token[idx_for_short_seq+1:idx_for_short_seq+1+block_size], torch.tensor([tokenizer.pad_token_id] * diff)], dim=0)
            attn_ = torch.cat([encodings['attention_mask'][i][idx_for_short_seq:idx_for_short_seq+block_size], torch.tensor([0] * diff)], dim=0)
            x.append(x_)
            y.append(y_)
            attn_mask.append(attn_)

    dataset = Dataset.from_dict({"input_ids": x, "attention_mask": attn_mask, "labels": y})
    return dataset

def evaluate(model, val_dataset, criterion, **kargs):
    eval_batch_size = kargs.get("eval_batch_size", 32)
    device = kargs.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    val_dataloader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False)

    model.eval()
    model.to(device)

    val_loss = 0.0
    val_correct, val_samples = 0, 0

    with torch.no_grad():
        for batch in val_dataloader:
            X_batch = batch['inputs'].to(device)
            y_batch = batch['labels'].to(device)

            y_batch_pred = model(X_batch)
            batch_loss = criterion(y_batch_pred, y_batch)
            val_loss += batch_loss.item()
            correct = ... # TODO
            val_correct += correct.sum().item()
            val_samples += len(batch)

    val_loss /= len(val_dataloader)
    acc = float(val_correct) / val_samples

    return {"val_loss": val_loss, "val_acc": acc}


def train(**kargs):
    batch_size = kargs.get("batch_size", 32)
    epochs = kargs.get("epoch", 5)
    lr = kargs.get("learning_rate", 6e-4)
    wd = kargs.get("weight_decay", 1e-1)
    betas = kargs.get("betas", (0.9, 0.95))
    device = kargs.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model_name = kargs.get("model_name", None)
    cache_dir = kargs.get("cache_dir", None)
    verbose = kargs.get("verbose", True)

    if model_name is None or cache_dir is None:
        model = GPT(GPTConfig(n_layer=1))
    else:
        model = get_llm(model_name, cache_dir, device)
    model.to(device)

    # TODO instead build a dataset in train.py, it is better that the dataset exists and load it from somewhere
    dataset = build_dataset()
    split_dataset = dataset.train_test_split(test_size=0.1)
    train_dataset = split_dataset["train"]
    train_dataset = train_dataset.shuffle(seed=seed)
    val_dataset = split_dataset["test"]
    batched_train_dataset = train_dataset.batch(batch_size=batch_size)

    criterion = kargs.get("criterion", nn.CrossEntropyLoss())
    optimizer = model.configure_optimizers(
        weight_decay=wd,
        learning_rate=lr,
        betas=betas,
        device_type=device
    )

    setup_and_hyperparameters = {
        "criterion": criterion.__class__.__name__,
        "optimizer": optimizer.__class__.__name__,
        "batch_size": batch_size,
        "epoch": epochs,
        "learning_rate": lr,
        "weight_decay": wd,
        "betas": betas
    }

    train_metrics = []
    val_metrics = []

    for epo in range(epochs):
        model.train()
        train_correct, train_samples = 0, 0
        train_epoch_loss = 0.0
        for batch in batched_train_dataset:
            X_batch = torch.tensor(batch['input_ids']).to(device)
            y_batch = torch.tensor(batch['labels']).to(device)
            attention_mask = torch.tensor(batch['attention_mask']).to(device)

            optimizer.zero_grad()
            y_batch_pred = model(X_batch)
            batch_loss = criterion(y_batch_pred, y_batch)
            batch_loss.backward()
            optimizer.step()

            train_epoch_loss += batch_loss.item()
            correct = ... # TODO
            train_correct += correct.sum().item()
            train_samples += len(batch)

        train_epoch_loss /= len(train_dataset)
        train_acc = float(train_correct) / train_samples
        train_metrics.append({"train_loss": train_epoch_loss, "train_acc": train_acc})

        # val loss and acc in a single epoch
        eval_metrics = evaluate(model, val_dataset, criterion, eval_batch_size=32, device=device)
        val_metrics.append(eval_metrics)

        if verbose:
            print(
                "Epoch: %.d, Train Loss: %.4f, Train Acc: %.4f, "
                "Val Loss: %.4f, Val Acc: %.4f" % (
                    epo + 1,
                    train_epoch_loss,
                    train_acc,
                    eval_metrics["val_loss"],
                    eval_metrics["val_acc"]
                )
            )

    new_model_name = ""
    for value in setup_and_hyperparameters.values():
        new_model_name += f"_{value}"
    new_model_name = new_model_name[1:]
    model.save_pretrained(cache_dir+"/"+new_model_name)

    return setup_and_hyperparameters, train_metrics, val_metrics