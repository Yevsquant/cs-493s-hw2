import argparse
import torch

from model import GPT, GPTConfig


def sample(model: GPT, prompt: str, stoi: dict[str, int], itos: dict[int, str], device) -> str:
    model.eval()
    with torch.no_grad():
        input_token = [stoi[i] for i in prompt]
        x = input_token + [stoi['<pad>']] * (model.config.block_size - len(input_token))
        x = torch.tensor(x, dtype=torch.long, device=device)
        x = x.unsqueeze(0)
        logits = model(x)
        pred = logits.argmax(-1).view(-1)
        pred = pred[len(input_token):]
        matches = (pred == stoi['\n']).nonzero(as_tuple=True)[0]
        first_index = -1
        if len(matches) > 0:
            first_index = matches[0].item()
        pred = pred[:first_index]
        out = "".join(itos[int(i.item())] for i in pred)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference with a trained model")
    parser.add_argument("--ckpt", type=str, default="ckpt_task23/model_seed=42_p=97.pt", help="Path to checkpoint pt file")
    parser.add_argument("--prompt", type=str, default="11/80=", help="Starting text")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location=args.device)
    config = GPTConfig(**ckpt["config"])
    model = GPT(config)
    model.load_state_dict(ckpt["model"])
    model.to(args.device)
    stoi = ckpt["stoi"]
    itos = {int(k): v for k, v in ckpt["itos"].items()} if isinstance(ckpt["itos"], dict) else ckpt["itos"]

    result = sample(model, args.prompt, stoi, itos, args.device)
    print(args.prompt + result)


if __name__ == "__main__":
    main()
