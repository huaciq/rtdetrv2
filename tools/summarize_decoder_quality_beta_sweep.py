"""Summarize decoder-quality inference diagnostics for five beta values."""

import argparse
import json
from pathlib import Path


BETAS = ('0', '0.1', '0.25', '0.5', '1')
METRICS = (
    'AP', 'AP50', 'AP75', 'APs', 'APm', 'APl',
    'AR100', 'ARs', 'ARm', 'ARl', 'AR75',
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'root', type=Path,
        help='Directory containing beta_0, beta_0.1, beta_0.25, beta_0.5, beta_1')
    parser.add_argument(
        '--checkpoint', type=str, default=None,
        help='Checkpoint path recorded as metadata without modifying it')
    args = parser.parse_args()

    runs = []
    for beta in BETAS:
        run_dir = args.root / f'beta_{beta}'
        with (run_dir / 'evaluation_metrics.json').open(
                encoding='utf-8') as file:
            metrics = json.load(file)
        with (run_dir / 'query_diagnosis' /
              'decoder_quality_alignment.json').open(
                  encoding='utf-8') as file:
            alignment = json.load(file)
        runs.append({
            'beta': float(beta),
            'metrics': {name: metrics[name] for name in METRICS},
            'decoder_quality_alignment': alignment,
        })

    summary = {
        'diagnostic': 'decoder quality beta sweep',
        'checkpoint': args.checkpoint,
        'formula': (
            'sigmoid(final_logits) * '
            'sigmoid(final_quality_logits) ** beta'),
        'beta_zero_semantics': 'classification score only',
        'runs': runs,
    }
    output_path = args.root / 'decoder_quality_beta_sweep.json'
    with output_path.open('w', encoding='utf-8') as file:
        json.dump(summary, file, indent=2)

    columns = ('beta', *METRICS, 'pearson', 'spearman')
    print('\t'.join(columns))
    for run in runs:
        alignment = run['decoder_quality_alignment']
        values = {
            'beta': run['beta'],
            **run['metrics'],
            'pearson': alignment['pearson'],
            'spearman': alignment['spearman'],
        }
        print('\t'.join(f'{values[name]:.6f}' for name in columns))
    print(f'JSON: {output_path}')


if __name__ == '__main__':
    main()
