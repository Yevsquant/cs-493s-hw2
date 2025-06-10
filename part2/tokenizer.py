import string
import torch
import os
import json

chars = list("0123456789+-/= ") # str to list, not sure if " " is really needed
vocab = {ch: i for i, ch in enumerate(chars)}
vocab['<pad>'] = len(vocab) + 10
vocab['<bos>'] = len(vocab) + 10
vocab['<eos>'] = len(vocab) + 10

inv_vocab = {i:ch for ch, i in vocab.items()}

class BinaryOpsTokenizer:
    def __init__(self, vocab, inv_vocab):
        self.vocab = vocab
        self.inv_vocab = inv_vocab
        self.pad_token = '<pad>'
        self.bos_token = '<bos>'
        self.eos_token = '<eos>'
        self.pad_token_id = vocab[self.pad_token]
        self.bos_token_id = vocab[self.bos_token]
        self.eos_token_id = vocab[self.eos_token]
    
    def encode(self, chs, add_special_tokens=True):
        token_ids = []
        if add_special_tokens:
            token_ids.append(self.vocab[self.bos_token])
        token_ids.extend([self.vocab[c] for c in chs])
        if add_special_tokens:
            token_ids.append(self.vocab[self.eos_token])
        return token_ids
    
    def decode(self, token_ids, skip_special_tokens=True):
        chs = [self.inv_vocab[i.item() if isinstance(i, torch.Tensor) else i] for i in token_ids]
        if not skip_special_tokens:
            return ''.join(chs)
        start, end = 0, len(chs)-1
        while chs[start] in {self.bos_token, self.pad_token, self.eos_token}:
            start += 1
        while chs[end] in {self.bos_token, self.pad_token, self.eos_token}:
            end -= 1
        chs = chs[start:end+1]
        return ''.join(chs)
    
    def __call__(self, chs_list, padding=None, truncation=False, max_length=None):
        """
        Always return tensor
        Example chs: "1+2=3"
        """
        if isinstance(chs_list, str):
            chs_list = [chs_list]
        
        all_token_ids = []
        for chs in chs_list:
            ids = self.encode(chs)
            if max_length is not None:
                ids = ids[:max_length]
            all_token_ids.append(ids) # len(ids) <= max_length
        
        # Padding
        padded, attention_masks = None, None
        if padding == "max_length" and max_length is not None:
            padded = []
            attention_masks = []
            for ids in all_token_ids:
                pad_len = max_length - len(ids)
                padded_ids, attn_mask = None, None
                if pad_len > 0:
                    padded_ids = ids + [self.pad_token_id] * pad_len
                    attn_mask = [1] * len(ids) + [0] * pad_len
                else:
                    padded_ids = ids
                    attn_mask = [1] * len(ids)
                padded.append(padded_ids[:max_length])
                attention_masks.append(attn_mask[:max_length])
        else:
            padded = all_token_ids
            attention_masks = [[1] * len(ids) for ids in all_token_ids]

        input_ids = torch.tensor(padded, dtype=torch.long)
        attention_mask = torch.tensor(attention_masks, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
    
    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        config_path = os.path.join(save_directory, "tokenizer_config.json")
        vocab_path = os.path.join(save_directory, "vocab.json")

        # Save any config info (optional)
        with open(config_path, "w") as f:
            json.dump({"tokenizer_class": self.__class__.__name__}, f)

        # Save vocab or internal state (you must define how your tokenizer serializes)
        with open(vocab_path, "w") as f:
            json.dump(self.vocab, f)  # assuming self.vocab is a dict

    @classmethod
    def from_pretrained(cls, load_directory):
        vocab_path = os.path.join(load_directory, "vocab.json")

        with open(vocab_path, "r") as f:
            vocab = json.load(f)

        return cls(vocab=vocab)

# test the tokenizer   
if __name__ == '__main__':
    tkn = BinaryOpsTokenizer(vocab, inv_vocab)
    expr = "12+7=19"
    exprs = ["12+7=19","12+7=19","12+7=9"]
    block_size = 12
    tokens = tkn(exprs)
    print("Encoded:", tokens)
    decoded = tkn.decode(tokens["input_ids"][0])
    print("Decoded:", decoded)
