from itertools import chain

import torch
import torch.nn.functional as F
from datasets import Dataset, DatasetDict, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

from model import GPT, GPTConfig

# Setup
tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
block_size = 8
model_config = GPTConfig(
    block_size = block_size,
    vocab_size = tokenizer.vocab_size,
    n_layer = 1,
    #n_head = 6,
    #n_embd = 384,
    dropout = 0.0
)
model_cache_dir = "models"
model_checkpoints = "models/checkpoints"
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_steps = 250
dataset_cache_dir = "nampdn-ai/tiny-textbooks"
dataset = None # DatasetDict
seed = 42

# Hyper parameters
batch_size = 1 # 1 for overfit, 32
num_train_epochs = 500 # 200 for overfit, 10
learning_rate = 5e-4 # AdamW
weight_decay = 0.0 # overfit
betas = (0.9, 0.95)
train_and_val_ratio = 0.1

def tokenize_function(example):
    return tokenizer(example["text"])

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
    result["labels"] = result["input_ids"].copy()
    return result

# Overfit for sanity check
text = "I love machine learning"
def overfit_dataset():
    # Tokenize
    tokens = tokenizer(text, return_tensors="pt")
    inputs = tokens["input_ids"][0][:-1]
    labels = tokens["input_ids"][0][1:].clone()
    labels[0] = -100 # as required

    # Add batch dimension
    inputs = inputs.unsqueeze(0)
    labels = labels.unsqueeze(0)
    attention_mask = tokens["attention_mask"][0][:inputs.shape[1]].unsqueeze(0) # might drop

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
def get_dataset(sanity_check=False):
    global dataset
    if sanity_check: # do overfitting
        dataset = overfit_dataset()
    else:
        raw_dataset = load_dataset(dataset_cache_dir)
        if "validation" not in raw_dataset.keys():
            ds = raw_dataset['train'].train_test_split(test_size=train_and_val_ratio, seed=seed)
            ds = DatasetDict({
                "train": ds["train"],
                "validation": ds["test"],
                "test": raw_dataset["test"]
            })
        ds = ds.map(tokenize_function, batched=True, remove_columns=["text"])
        dataset = ds.map(group_texts, batched=True)

class GPTTrainer(Trainer):
    """
    def create_optimizer(self):
        optimizer = model.configure_optimizers(
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            betas=betas,
            device_type=device
        )
        return optimizer
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        logits = model(inputs["input_ids"]) # **inputs
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        return (loss, logits) if return_outputs else loss

def train(**kargs):
    # Model for training
    global model
    model_name = kargs.get("model_name", None)
    if model_name is None or model_cache_dir is None:
        model = GPT(model_config)
    else:
        model = get_llm(model_name)
    model.to(device)

    # Dataset for training
    sanity_check = kargs.get("sanity_check", False)
    get_dataset(sanity_check)

    training_args = TrainingArguments(
        output_dir=model_checkpoints,
        overwrite_output_dir=True,
        per_device_train_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        logging_steps=10,
        save_steps=save_steps,
        save_safetensors=False, # safetensors not support saving shared weights
        learning_rate=learning_rate,
        warmup_steps=0,
        weight_decay=weight_decay,
        report_to="none",
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = GPTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

# Move to inference, might ref from nanoGPT
def generate_greedy(model, tokenizer, prompt, max_new_tokens=50, temperature=1.0):
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
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        input_ids = torch.cat((input_ids, next_token), dim=1)

    return input_ids

if __name__ == "__main__":
    # Train the model
    train(sanity_check=True)

    model.eval()
    output = generate_greedy(model, tokenizer, "I love machine", max_new_tokens=5)
    print(tokenizer.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True))
