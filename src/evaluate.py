"""Clean / robust accuracy evaluation. Appends one row per run to results/results.csv."""
import argparse
import csv
import os
import time

import torch
from tqdm import tqdm

from src.attacks import get_attack
from src.data import get_loaders
from src.models import load_baseline, load_checkpoint


def evaluate(model, loader, attack=None, max_batches=None, device="cuda"):
    correct = total = 0
    for i, (x, y) in enumerate(tqdm(loader)):
        if max_batches and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        if attack is not None:
            x = attack(x, y)
        with torch.no_grad():
            correct += (model(x).argmax(1) == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--attack", default="none")
    p.add_argument("--eps", type=float, default=8)
    p.add_argument("--checkpoint", default=None, help="defended model; omit for baseline")
    p.add_argument("--max-batches", type=int, default=None, help="subset for quick runs")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_checkpoint(args.checkpoint, device=device) if args.checkpoint else load_baseline(device=device)
    _, test_loader = get_loaders()
    attack = None if args.attack == "none" else get_attack(args.attack, model, args.eps)

    start = time.time()
    acc = evaluate(model, test_loader, attack, args.max_batches, device)
    row = {"model": args.checkpoint or "baseline", "attack": args.attack, "eps_255": args.eps,
           "accuracy": round(acc, 2), "seconds": round(time.time() - start, 1)}
    print(row)

    os.makedirs("results", exist_ok=True)
    path = "results/results.csv"
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if new:
            w.writeheader()
        w.writerow(row)


if __name__ == "__main__":
    main()
