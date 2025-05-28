import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

# Load custom models
def get_llm(model_name, cache_dir="models", device="cpu"):
    model_path = cache_dir + "/" + model_name
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        # torch_dtype=torch.float16
    )
    model.to(device)
    model.eval()
    return tokenizer, model

def generate(tokenizer, model, prompt, device, max_new_tokens):
    inputs = tokenizer(prompt, return_tensor="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            nax_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = get_llm(args.model_name, args.cache_dir, device)
    result = generate(tokenizer, model, args.prompt, device, args.max_tokens)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model")
    parser.add_argument("--cache_dir", type=str, required=True, help="Path to saved model")
    parser.add_argument("--prompt", type=str, required=True, help="Text for generating")
    parser.add_argument("--max_tokens", type=int, default=50, help="Number of tokens to generate")

    args = parser.parse_args()

    main(args)