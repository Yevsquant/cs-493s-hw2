from itertools import chain
from typing import Optional, Any, Dict

import torch
import torch.nn.functional as F
from torch.optim import Adam
from datasets import Dataset, DatasetDict, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

from model import GPT, GPTConfig
from part2.tokenizer import BinaryOpsTokenizer, vocab, inv_vocab

# Setup
tokenizer = BinaryOpsTokenizer(vocab, inv_vocab)
block_size = 32
model_config = GPTConfig(
    block_size = block_size,
    vocab_size = 512,
    n_layer = 2,
    n_head = 4,
    n_embd = 128,
    dropout = 0.1
)
model_cache_dir = "models"
model_checkpoints = "models/mod_checkpoints_warmup"
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_steps = 1000
dataset_cache_dir = None
dataset = None # DatasetDict
seeds = [42]
ps = [113] # 97, 113

# Hyper parameters
batch_size = 64 # 1 for overfit, 32
num_train_epochs = 100
learning_rate = 3e-4 # Adam
weight_decay = 0.0 # overfit
betas = (0.9, 0.98)
train_and_val_ratio = 0.1

def tokenize_function(example):
    t = tokenizer(example["text"], padding='max_length', max_length=16, truncation=True)
    return t

def group_texts(examples):
    global block_size
    concatenated = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated["input_ids"]) # should be same as concatenated[list(examples.keys())[0]]
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated.items()
    }
    result["labels"] = list(result["input_ids"])
    return result

# Overfit for sanity check
text = "I love machine learning" + tokenizer.eos_token
def overfit_dataset():
    # Tokenize
    tokens = tokenizer(text, padding='max_length', max_length=block_size, truncation=True, return_tensors="pt")
    inputs = tokens["input_ids"][0].clone()
    labels = tokens["input_ids"][0].clone()

    # Add batch dimension
    inputs = inputs.unsqueeze(0)
    labels = labels.unsqueeze(0)
    attention_mask = tokens["attention_mask"][0].unsqueeze(0) # might drop if no needed
    # labels[attention_mask == 0] = -100

    # Create a dataset
    ds = Dataset.from_dict({
        "input_ids": [inputs[0]],
        "attention_mask": [attention_mask[0]],
        "labels": [labels[0]],
    })

    dataset = DatasetDict({
        "train": ds,
        "validation": ds,
        "test": ds
    })
    return dataset

# Load models
def get_llm(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_cache_dir + "/" + model_name,
        # torch_dtype=torch.float16
    )
    model.to(device)
    return model

# Load dataset
def get_dataset(seed, p, sanity_check=False):
    global dataset
    if sanity_check: # do overfitting
        dataset = overfit_dataset()
    else:
        raw_dataset = load_dataset("text", data_files={"train": f"part2/AlgorithmicTasks/train_p{p}.txt", "validation": f"part2/AlgorithmicTasks/val_p{p}.txt"})
        if "validation" not in raw_dataset.keys():
            ds = raw_dataset['train'].train_test_split(test_size=train_and_val_ratio, seed=seed)
            ds = DatasetDict({
                "train": ds["train"],
                "validation": ds["test"],
                "test": raw_dataset["test"]
            })
        dataset = raw_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
        dataset = dataset.map(group_texts, batched=True)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits = torch.tensor(logits)
    labels = torch.tensor(labels)
    eq_token_id = vocab['=']
    ignore_index = -100
    
    # Mask labels (same as in compute_loss)
    masked_labels = labels.clone()
    for i in range(labels.size(0)):
        eq_mask = (labels[i] == eq_token_id)
        eq_indices = eq_mask.nonzero(as_tuple=True)[0]
        if len(eq_indices) > 0:
            cutoff = eq_indices[0].item()
            masked_labels[i, :cutoff+1] = ignore_index

    # Shift for next-token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = masked_labels[..., 1:].contiguous()

    # Flatten tensors
    shift_logits = shift_logits.view(-1, shift_logits.shape[-1])
    shift_labels = shift_labels.view(-1)

    # Create mask for valid tokens
    mask = shift_labels != ignore_index
    if mask.sum() == 0:  # Handle empty valid tokens
        return {"accuracy": 0.0}

    # Calculate accuracy
    valid_logits = shift_logits[mask]
    valid_labels = shift_labels[mask]
    preds = valid_logits.argmax(dim=-1)
    acc = (preds == valid_labels).float().mean().item()
    return {"accuracy": acc}

class GPTTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._train_logits = []
        self._train_labels = []

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        if self.optimizer is None:
            self.optimizer = Adam(
                self.model.parameters(),
                lr=learning_rate,
                betas=betas,
                weight_decay=weight_decay,
            )
        self.lr_scheduler = self.create_scheduler(num_training_steps, self.optimizer)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        input_ids = torch.tensor(inputs["input_ids"])
        attention_mask = torch.tensor(inputs["attention_mask"])
        labels = torch.tensor(inputs.pop("labels"))
        eq = vocab['=']
        ignore_index = -100
        masked_labels = labels.clone()
        for i in range(input_ids.size(0)):
            eq_idx = (input_ids[i] == eq).nonzero(as_tuple=True)[0]
            if len(eq_idx) > 0:
                cutoff = eq_idx[0].item()
                masked_labels[i, :cutoff + 1] = ignore_index
        # logits = model(inputs["input_ids"]) # **inputs
        logits = model(input_ids, attention_mask) # **inputs
        # Shift logits and labels for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = masked_labels[..., 1:].contiguous()
        
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=ignore_index
        )

        # Save logits and labels for logging train accuracy
        self._train_logits.append(logits.detach().cpu())
        self._train_labels.append(labels.detach().cpu())

        return (loss, (logits, labels)) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        model.eval()
        with torch.no_grad():
            input_ids = torch.tensor(inputs["input_ids"])
            attention_mask = torch.tensor(inputs["attention_mask"])
            labels = torch.tensor(inputs.get("labels"))

            logits = model(input_ids, attention_mask)

            if labels is not None:
                eq = vocab['=']
                ignore_index = -100
                masked_labels = labels.clone()
                for i in range(input_ids.size(0)):
                    eq_idx = (input_ids[i] == eq).nonzero(as_tuple=True)[0]
                    if len(eq_idx) > 0:
                        cutoff = eq_idx[0].item()
                        masked_labels[i, :cutoff + 1] = ignore_index

                # Compute loss
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = masked_labels[..., 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=ignore_index
                )
            else:
                loss = None

        if prediction_loss_only:
            return (loss, None, None)

        return (loss, logits, labels)

    # custom log for train acc
    def log(self, logs: Dict[str, float], iterator: Optional[Any] = None) -> None:
        if len(self._train_logits) > 0 and len(self._train_labels) > 0:
            logits = torch.cat(self._train_logits, dim=0)
            labels = torch.cat(self._train_labels, dim=0)

            eq_token_id = vocab['=']
            ignore_index = -100

            masked_labels = labels.clone()
            for i in range(labels.size(0)):
                eq_idx = (labels[i] == eq_token_id).nonzero(as_tuple=True)[0]
                if len(eq_idx) > 0:
                    cutoff = eq_idx[0].item()
                    masked_labels[i, :cutoff + 1] = ignore_index

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = masked_labels[..., 1:].contiguous()
            mask = shift_labels != ignore_index

            if mask.sum() > 0:
                preds = shift_logits[mask].argmax(dim=-1)
                acc = (preds == shift_labels[mask]).float().mean().item()
                logs["train_accuracy"] = acc
                if self.state.log_history:
                    self.state.log_history[-1]["train_accuracy"] = acc

            # Clear saved logits and labels after logging to avoid memory issues
            self._train_logits = []
            self._train_labels = []

        super().log(logs, iterator)


def train(**kargs):
    for seed in seeds:
        for p in ps:

            # # Model for training
            # global model
            # model_name = kargs.get("model_name", None)
            # if model_name is None or model_cache_dir is None:
            #     model = GPT(model_config)
            # else:
            #     model = get_llm(model_name)
            # model.to(device)
            global model
            model = GPT(model_config)
            model.to(device)

            # Dataset for training
            sanity_check = kargs.get("sanity_check", False)
            get_dataset(seed, p, sanity_check)

            training_args = TrainingArguments(
                output_dir=f"{model_checkpoints}/{seed}_{p}",
                overwrite_output_dir=True,
                per_device_train_batch_size=batch_size,
                num_train_epochs=num_train_epochs,
                eval_strategy="steps",
                eval_steps=10,
                logging_steps=10,
                save_steps=save_steps,
                save_safetensors=False, # safetensors not support saving shared weights
                save_strategy="no",
                learning_rate=learning_rate,
                warmup_steps=5,
                weight_decay=weight_decay,
                report_to="tensorboard",
                logging_dir=f"./logs_warmup/{seed}_{p}",
                # report_to="none",
                remove_unused_columns=False,
            )

            # seqs are guaranteed to be same length
            data_collator = lambda data: {
                key: torch.stack([torch.tensor(f[key]) for f in data])
                for key in data[0]
            }

            trainer = GPTTrainer(
                model=model,
                args=training_args,
                train_dataset=dataset['train'],
                eval_dataset=dataset['validation'],
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
            )

            trainer.train()

# Move to inference, might ref from nanoGPT
def generate_greedy(model, tokenizer, prompt, max_new_tokens=50, temperature=1.0, top_k=None):
    model.eval()
    device = next(model.parameters()).device
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    for _ in range(max_new_tokens):
        input_cond = input_ids
        if input_ids.size(1) > model.config.block_size:
            input_cond = input_ids[:, -model.config.block_size:]

        with torch.no_grad():
            logits = model(input_cond)
            logits = logits[:, -1, :] / temperature

            # from nanoGPT
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            if next_token.item() == tokenizer.eos_token_id:
                # break
                return input_ids

        input_ids = torch.cat((input_ids, next_token), dim=1)

    return input_ids

if __name__ == "__main__":
    # Train the model
    train(sanity_check=False)

    # model.eval()
    # output = generate_greedy(model, tokenizer, "I love machine", max_new_tokens=10, top_k=3)
    # print(tokenizer.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True))
