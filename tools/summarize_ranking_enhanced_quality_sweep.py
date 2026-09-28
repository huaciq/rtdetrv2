"""Validate beta=0 and summarize ranking-enhanced probe evaluations."""

import argparse
import json
from pathlib import Path


BETAS = ('0', '0.05', '0.1', '0.25', '0.5')
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
        help='Directory containing beta_0, beta_0.05, ..., beta_0.5')
    parser.add_argument(
        '--baseline-metrics', type=Path, required=True,
        help='evaluation_metrics.json from the original QAQS checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--lambda-rank', type=float, required=True)
    parser.add_argument('--tolerance', type=float, default=1e-12)
    parser.add_argument('--verify-beta-zero-only', action='store_true')
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
            'beta=0 does not recover original QAQS evaluation; stop and '
            f'investigate: {details}')
    print('beta=0 original QAQS recovery: PASS')
    if args.verify_beta_zero_only:
        return

    runs = []
    for beta in BETAS:
        run_dir = args.root / f'beta_{beta}'
        metrics = load_json(run_dir / 'evaluation_metrics.json')
        alignment = load_json(
            run_dir / 'query_diagnosis' /
            'class_conditioned_quality_alignment.json')
        runs.append({
            'beta': float(beta),
            'metrics': {name: metrics[name] for name in METRICS},
            'quality_alignment': alignment,
        })

    summary = {
        'diagnostic': 'ranking-enhanced class-conditioned quality probe',
        'lambda_rank': args.lambda_rank,
        'checkpoint': args.checkpoint,
        'formula': (
            'sigmoid(final_logits[i,c]) * '
            'sigmoid(quality_logits[i,c]) ** beta'),
        'beta_zero_check': {
            'passed': True,
            'tolerance': args.tolerance,
            'baseline_metrics': str(args.baseline_metrics),
            'absolute_differences': differences,
        },
        'runs': runs,
    }
    output_path = args.root / 'ranking_enhanced_quality_beta_sweep.json'
    with output_path.open('w', encoding='utf-8') as file:
        json.dump(summary, file, indent=2)

    columns = (
        'beta', *METRICS, 'top100_spearman', 'iou_gt_0.5_spearman',
        'iou_gt_0.75_spearman', 'same_query_rate', 'mean_regret',
        'median_regret', 'p90_regret')
    print('\t'.join(columns))
    for run in runs:
        alignment = run['quality_alignment']
        conditional = alignment['conditional_correlations']
        ranking = alignment['per_image_per_class_ranking']
        values = {
            'beta': run['beta'],
            **run['metrics'],
            'top100_spearman': alignment['spearman'],
            'iou_gt_0.5_spearman': (
                conditional['true_quality > 0.5']['spearman']),
            'iou_gt_0.75_spearman': (
                conditional['true_quality > 0.75']['spearman']),
            'same_query_rate': ranking['same_query_rate'],
            'mean_regret': ranking['mean_iou_regret'],
            'median_regret': ranking['median_iou_regret'],
            'p90_regret': ranking['p90_iou_regret'],
        }
        print('\t'.join(format_value(values[name]) for name in columns))
    print(f'JSON: {output_path}')


if __name__ == '__main__':
    main()
