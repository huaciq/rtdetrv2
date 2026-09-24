"""Validate beta=0 and summarize the final-quality-probe beta sweep."""

import argparse
import json
from pathlib import Path


BETAS = ('0', '0.1', '0.25', '0.5', '1')
METRICS = (
    'AP', 'AP50', 'AP75', 'APs', 'APm', 'APl',
    'AR100', 'ARs', 'ARm', 'ARl', 'AR75',
)


def load_json(path):
    with path.open(encoding='utf-8') as file:
        return json.load(file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'root', type=Path,
        help='Directory containing beta_0, beta_0.1, beta_0.25, beta_0.5, beta_1')
    parser.add_argument(
        '--baseline-metrics', type=Path, required=True,
        help='evaluation_metrics.json from the original QAQS checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--tolerance', type=float, default=1e-12)
    parser.add_argument(
        '--verify-beta-zero-only', action='store_true',
        help='Check beta=0 against QAQS and exit before loading other betas')
    args = parser.parse_args()

    baseline = load_json(args.baseline_metrics)
    beta_zero_raw = load_json(
        args.root / 'beta_0' / 'evaluation_metrics.json')
    beta_zero = {name: beta_zero_raw[name] for name in METRICS}
    differences = {
        name: abs(beta_zero[name] - baseline[name]) for name in METRICS
    }
    failed = {
        name: difference for name, difference in differences.items()
        if difference > args.tolerance
    }
    if failed:
        details = ', '.join(
            f'{name} diff={difference:.12g}'
            for name, difference in failed.items())
        raise RuntimeError(
            'beta=0 does not recover the original QAQS metrics; stop the '
            f'experiment and investigate: {details}')
    print('beta=0 QAQS recovery: PASS')
    if args.verify_beta_zero_only:
        return

    runs = []
    for beta in BETAS:
        run_dir = args.root / f'beta_{beta}'
        metrics = load_json(run_dir / 'evaluation_metrics.json')
        alignment = load_json(
            run_dir / 'query_diagnosis' /
            'decoder_quality_alignment.json')
        runs.append({
            'beta': float(beta),
            'metrics': {name: metrics[name] for name in METRICS},
            'decoder_quality_alignment': alignment,
        })

    summary = {
        'diagnostic': 'final decoder quality probe beta sweep',
        'checkpoint': args.checkpoint,
        'formula': (
            'sigmoid(final_logits) * '
            'sigmoid(final_quality_probe_logits) ** beta'),
        'quality_target': (
            'predicted-class-aware max final-box IoU; zero when the '
            'predicted class has no GT'),
        'beta_zero_check': {
            'passed': True,
            'tolerance': args.tolerance,
            'baseline_metrics': str(args.baseline_metrics),
            'absolute_differences': differences,
        },
        'runs': runs,
    }
    output_path = args.root / 'final_quality_probe_beta_sweep.json'
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
