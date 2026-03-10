"""Prepare training JSON from Stage1 pickle file.
Usage:
    python -m scripts.prepare_training_data --pkl offline_dataset.pkl --output training_data_with_vstar.json
"""
import argparse
from datasets import load_dataset
from pag_a_star_po.data_prep import prepare_training_data, verify_data_format


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl', default='offline_dataset.pkl')
    parser.add_argument('--output', default='training_data_with_vstar.json')
    args = parser.parse_args()

    train_dataset = load_dataset("qwedsacf/competition_math")['train']
    train_proc = train_dataset.map(lambda x: {"problem": f"{x['problem']}\n", "solution": x['solution']}, remove_columns=[c for c in train_dataset.column_names if c not in ['problem','solution']])
    n = prepare_training_data(args.pkl, train_proc.select(range(100)), output_path=args.output)
    print(f"Prepared {n} training samples -> {args.output}")
    ok = verify_data_format(args.output)
    print(f"Verify OK: {ok}")

if __name__ == '__main__':
    cli()
