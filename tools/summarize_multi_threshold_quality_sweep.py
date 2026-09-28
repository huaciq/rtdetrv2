"""Validate beta=0 and summarize multi-threshold quality evaluations."""

import argparse
import json
from pathlib import Path


BETAS = ('0', '0.05', '0.1', '0.25', '0.5', '1')
METRICS = (
    'AP', 'AP50', 'AP75', 'APs', 'APm', 'APl',
    'AR100', 'ARs', 'ARm', 'ARl', 'AR75',
)


def load_json(path):
    with path.open(encoding='utf-8') as file:
        return json.load(file)


def format_value(value):
    return 'N/A' if value is None else f'{value:.6f}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'root', type=Path,
        help='Directory containing beta_0, beta_0.05, ..., beta_1')
    parser.add_argument(
        '--baseline-metrics', type=Path, required=True,
        help='evaluation_metrics.json from the original QAQS checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--tolerance', type=float, default=1e-12)
    parser.add_argument(
        '--verify-beta-zero-only', action='store_true',
        help='Check beta=0 and stop before loading the other runs')
    args = parser.parse_args()

    baseline = load_json(args.baseline_metrics)
    beta_zero_raw = load_json(
        args.root / 'beta_0' / 'evaluation_metrics.json')
    beta_zero = {name: beta_zero_raw[name] for name in METRICS}
    differences = {
        name: abs(beta_zero[name] - baseline[name]) for name in METRICS
    }
    failures = {
        name: difference for name, difference in differences.items()
        if difference > args.tolerance
    }
    if failures:
        details = ', '.join(
            f'{name} diff={difference:.12g}'
            for name, difference in failures.items())
        raise RuntimeError(
            'beta=0 does not recover original QAQS evaluation; stop the '
            f'sweep and investigate: {details}')
    print('beta=0 original QAQS recovery: PASS')
    if args.verify_beta_zero_only:
        return

    runs = []
    for beta in BETAS:
        run_dir = args.root / f'beta_{beta}'
        metrics = load_json(run_dir / 'evaluation_metrics.json')
        alignment = load_json(
            run_dir / 'query_diagnosis' /
            'multi_threshold_quality_alignment.json')
        runs.append({
            'beta': float(beta),
            'metrics': {name: metrics[name] for name in METRICS},
            'quality_alignment': alignment,
        })

    summary = {
        'diagnostic': (
            'frozen multi-threshold class-conditioned final quality probe'),
        'checkpoint': args.checkpoint,
        'thresholds': [0.50, 0.55, 0.60, 0.65, 0.70,
                       0.75, 0.80, 0.85, 0.90, 0.95],
        'formula': (
            'sigmoid(final_logits[i,c]) * '
            'mean_k(sigmoid(quality_logits[i,c,k])) ** beta'),
        'selection': (
            'flatten [query,class] scores and apply global Top-K exactly as '
            'RTDETRPostProcessor'),
        'beta_zero_check': {
            'passed': True,
            'tolerance': args.tolerance,
            'baseline_metrics': str(args.baseline_metrics),
            'absolute_differences': differences,
        },
        'runs': runs,
    }
    output_path = args.root / 'multi_threshold_quality_beta_sweep.json'
    with output_path.open('w', encoding='utf-8') as file:
        json.dump(summary, file, indent=2)

    columns = ('beta', *METRICS, 'mean_auroc', 'pearson', 'spearman')
    print('\t'.join(columns))
    for run in runs:
        alignment = run['quality_alignment']
        values = {
            'beta': run['beta'],
            **run['metrics'],
            'mean_auroc': alignment['mean_auroc'],
            'pearson': alignment['pearson'],
            'spearman': alignment['spearman'],
        }
        print('\t'.join(format_value(values[name]) for name in columns))
    print(f'JSON: {output_path}')


if __name__ == '__main__':
    main()
