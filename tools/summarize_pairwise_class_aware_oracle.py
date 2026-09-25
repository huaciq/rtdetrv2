"""Validate and summarize pair-wise class-aware Oracle evaluations."""

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
        help='Original QAQS evaluation_metrics.json')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--tolerance', type=float, default=1e-12)
    parser.add_argument(
        '--verify-beta-zero-only', action='store_true',
        help='Check beta=0 and stop before reading the other runs')
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
            'beta=0 does not strictly recover original QAQS evaluation; '
            f'stop the Oracle sweep: {details}')
    print('beta=0 original QAQS recovery: PASS')
    if args.verify_beta_zero_only:
        return

    runs = []
    for beta in BETAS:
        metrics = load_json(
            args.root / f'beta_{beta}' / 'evaluation_metrics.json')
        runs.append({
            'beta': float(beta),
            'metrics': {name: metrics[name] for name in METRICS},
        })

    summary = {
        'diagnostic': (
            'PAIR-WISE CLASS-AWARE ORACLE DIAGNOSTIC; '
            'NOT DEPLOYABLE MODEL PERFORMANCE'),
        'checkpoint': args.checkpoint,
        'formula': (
            'sigmoid(final_logits[i,c]) * '
            'max_iou(final_bbox[i], gt_of_class_c) ** beta'),
        'postprocessor_selection': (
            'flatten [num_queries, num_classes] scores, then global Top-K '
            'over query-class pairs'),
        'beta_zero_check': {
            'passed': True,
            'tolerance': args.tolerance,
            'baseline_metrics': str(args.baseline_metrics),
            'absolute_differences': differences,
        },
        'runs': runs,
    }
    output_path = args.root / 'pairwise_class_aware_oracle_summary.json'
    with output_path.open('w', encoding='utf-8') as file:
        json.dump(summary, file, indent=2)

    columns = ('beta', *METRICS)
    print('PAIR-WISE CLASS-AWARE ORACLE DIAGNOSTIC ONLY')
    print('NOT DEPLOYABLE MODEL PERFORMANCE')
    print('\t'.join(columns))
    for run in runs:
        values = {'beta': run['beta'], **run['metrics']}
        print('\t'.join(f'{values[name]:.6f}' for name in columns))
    print(f'JSON: {output_path}')


if __name__ == '__main__':
    main()
