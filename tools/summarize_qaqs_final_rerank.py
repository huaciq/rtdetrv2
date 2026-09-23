"""Summarize the four inference-only QAQS final re-ranking evaluations."""

import argparse
import json
from pathlib import Path


GAMMAS = ('0', '0.25', '0.5', '1')
METRICS = ('AP', 'AP50', 'AP75', 'APs', 'APm', 'APl',
           'AR100', 'ARs', 'ARm', 'ARl')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'root', type=Path,
        help='Directory containing gamma_0, gamma_0.25, gamma_0.5, gamma_1')
    parser.add_argument(
        '--oracle-diagnostic', action='store_true',
        help='Label the summary as a GT-assisted oracle diagnostic')
    parser.add_argument(
        '--class-aware-oracle-diagnostic', action='store_true',
        help='Label and name the strict predicted-class oracle summary')
    args = parser.parse_args()
    if args.oracle_diagnostic and args.class_aware_oracle_diagnostic:
        parser.error('Choose only one oracle diagnostic label')

    rows = []
    for gamma in GAMMAS:
        run_dir = args.root / f'gamma_{gamma}'
        with (run_dir / 'evaluation_metrics.json').open(
                encoding='utf-8') as file:
            coco = json.load(file)
        with (run_dir / 'query_diagnosis' /
              'final_score_alignment.json').open(encoding='utf-8') as file:
            alignment = json.load(file)
        row = {'gamma': float(gamma)}
        if args.oracle_diagnostic:
            row['diagnostic'] = 'ORACLE FINAL-IOU RE-RANKING'
        elif args.class_aware_oracle_diagnostic:
            row['diagnostic'] = (
                'PREDICTED-CLASS-AWARE ORACLE FINAL-IOU RE-RANKING')
        row.update({name: coco[name] for name in METRICS})
        row['pearson'] = alignment['pearson']
        row['spearman'] = alignment['spearman']
        row['correlation_samples'] = alignment['sample_count']
        rows.append(row)

    args.root.mkdir(parents=True, exist_ok=True)
    if args.class_aware_oracle_diagnostic:
        output_name = 'class_aware_oracle_summary.json'
    elif args.oracle_diagnostic:
        output_name = 'oracle_summary.json'
    else:
        output_name = 'summary.json'
    output_path = args.root / output_name
    with output_path.open('w', encoding='utf-8') as file:
        json.dump(rows, file, indent=2)

    columns = ('gamma', *METRICS, 'pearson', 'spearman')
    if args.oracle_diagnostic:
        print('ORACLE DIAGNOSTIC ONLY - NOT DEPLOYABLE MODEL PERFORMANCE')
    elif args.class_aware_oracle_diagnostic:
        print('PREDICTED-CLASS-AWARE ORACLE DIAGNOSTIC ONLY - '
              'NOT DEPLOYABLE MODEL PERFORMANCE')
    print('\t'.join(columns))
    for row in rows:
        print('\t'.join(f'{row[name]:.6f}' for name in columns))
    print(f'JSON: {output_path}')


if __name__ == '__main__':
    main()
