"""Validate identity scoring and summarize scalar quality score fusion."""

import argparse
import json
from pathlib import Path


METRICS = (
    'AP', 'AP50', 'AP75', 'APs', 'APm', 'APl',
    'AR100', 'ARs', 'ARm', 'ARl', 'AR75',
)

RUNS = (
    *((f'product_beta_{value}', 'product', {'beta': float(value)})
      for value in ('0.05', '0.1', '0.25', '0.5')),
    *((f'geometric_alpha_{value}', 'geometric', {'alpha': float(value)})
      for value in ('0.05', '0.1', '0.2', '0.3')),
    *((f'linear_alpha_{value}', 'linear', {'alpha': float(value)})
      for value in ('0.02', '0.05', '0.1', '0.2')),
    *((f'gated_tau_{tau}_beta_{beta}', 'gated_product',
       {'tau': float(tau), 'beta': float(beta)})
      for tau in ('0.05', '0.1', '0.2', '0.3')
      for beta in ('0.1', '0.25')),
)


def load_json(path):
    with path.open(encoding='utf-8') as file:
        return json.load(file)


def select_metrics(raw):
    return {name: raw[name] for name in METRICS}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument(
        '--baseline-metrics', type=Path, required=True,
        help='QAQS evaluation_metrics.json from the same hardware/protocol')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--tolerance', type=float, default=1e-12)
    parser.add_argument('--verify-identity-only', action='store_true')
    args = parser.parse_args()

    baseline = select_metrics(load_json(args.baseline_metrics))
    identity = select_metrics(load_json(
        args.root / 'identity' / 'evaluation_metrics.json'))
    identity_differences = {
        name: abs(identity[name] - baseline[name]) for name in METRICS
    }
    failures = {
        name: value for name, value in identity_differences.items()
        if value > args.tolerance
    }
    if failures:
        details = ', '.join(
            f'{name} diff={difference:.12g}'
            for name, difference in failures.items())
        raise RuntimeError(
            'Zero-weight identity scoring does not recover QAQS; stop the '
            f'sweep and investigate: {details}')
    print('zero-weight score fusion recovers QAQS baseline: PASS')
    if args.verify_identity_only:
        return

    runs = []
    for directory, method, parameters in RUNS:
        metrics = select_metrics(load_json(
            args.root / directory / 'evaluation_metrics.json'))
        deltas = {
            name: metrics[name] - baseline[name] for name in METRICS
        }
        runs.append({
            'directory': directory,
            'method': method,
            'parameters': parameters,
            'metrics': metrics,
            'delta_vs_qaqs': deltas,
        })

    runs.sort(
        key=lambda run: (
            run['metrics']['AP'],
            run['metrics']['AP75'],
            run['metrics']['APs']),
        reverse=True)
    top5 = runs[:5]
    summary = {
        'diagnostic': 'inference-only scalar class-conditioned score fusion',
        'checkpoint': args.checkpoint,
        'baseline_metrics_path': str(args.baseline_metrics),
        'baseline_metrics': baseline,
        'identity_check': {
            'passed': True,
            'tolerance': args.tolerance,
            'absolute_differences': identity_differences,
        },
        'ranking_order': 'AP descending, then AP75, then APs',
        'runs_sorted_by_ap': runs,
        'top5': top5,
    }
    output_path = args.root / 'score_fusion_sweep.json'
    with output_path.open('w', encoding='utf-8') as file:
        json.dump(summary, file, indent=2)

    print('\nAll score-fusion runs (sorted by AP):')
    header = ('rank', 'method', 'parameters', *METRICS)
    print('\t'.join(header))
    for rank, run in enumerate(runs, 1):
        row = (
            str(rank), run['method'], json.dumps(run['parameters']),
            *(f'{run["metrics"][name]:.6f}' for name in METRICS),
        )
        print('\t'.join(row))

    print('\nBest 5 configurations and gains over QAQS:')
    for rank, run in enumerate(top5, 1):
        metrics = run['metrics']
        delta = run['delta_vs_qaqs']
        print(
            f'{rank}. {run["method"]} {run["parameters"]}: '
            f'AP={metrics["AP"]:.6f} ({delta["AP"]:+.6f}), '
            f'AP75={metrics["AP75"]:.6f} ({delta["AP75"]:+.6f}), '
            f'APs={metrics["APs"]:.6f} ({delta["APs"]:+.6f})')
    print(f'JSON: {output_path}')


if __name__ == '__main__':
    main()
