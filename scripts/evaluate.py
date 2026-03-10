"""CLI wrapper to evaluate a model directory.
Usage:
    python -m scripts.evaluate --model_dir ./model_dir --n_samples 4 --device cpu
"""
import argparse
from pag_a_star_po.evaluate import evaluate_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--n_samples', type=int, default=4)
    parser.add_argument('--device', default=None)
    args = parser.parse_args()

    acc_t1, acc_final = evaluate_model(args.model_dir, n_samples=args.n_samples, device=args.device)
    print(f"MATH-500 Results: Acc@t1={acc_t1:.2f}%, Acc@final={acc_final:.2f}%, Improvement={acc_final-acc_t1:.2f}%")

if __name__ == '__main__':
    main()
