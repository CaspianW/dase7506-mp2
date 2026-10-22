"""Run fresh official prediction and scoring processes with external CPU timing."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--split', choices=['validation', 'test', 'audit'], required=True)
    parser.add_argument('--checkpoint', type=Path, default=root / 'checkpoint')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output directory.')
    args.output.mkdir(parents=True)
    env = os.environ.copy()
    for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        env[name] = '4'
    env.update(OMP_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    predictions, score = args.output / 'predictions.json', args.output / 'score.json'
    commands = [
        [sys.executable, '-B', str(root / 'src/predict.py'), '--method', str(root / 'src/submission.py'),
         '--checkpoint', str(args.checkpoint.resolve()), '--data-dir', str(args.data_dir.resolve()),
         '--split', args.split, '--output', str(predictions.resolve())],
        [sys.executable, '-B', str(root / 'src/score_predictions.py'), '--data-dir', str(args.data_dir.resolve()),
         '--split', args.split, '--predictions', str(predictions.resolve()), '--output', str(score.resolve())],
    ]
    assets = {p.name: {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(args.checkpoint.iterdir()) if p.is_file()}
    before = {'split': args.split, 'python': sys.version, 'platform': platform.platform(),
        'processor': platform.processor(), 'cpu_numeric_threads_limit': 4,
        'cuda_visible_devices': '', 'commands': commands, 'inference_assets': assets,
        'inference_asset_bytes': sum(v['bytes'] for v in assets.values()),
        'source_sha256': {p.name: sha(p) for p in sorted((root / 'src').glob('*.py'))}}
    (args.output / 'receipt.json').write_text(json.dumps(before, indent=2) + '\n', encoding='utf-8')
    timings = []
    for command in commands:
        start = time.perf_counter()
        subprocess.run(command, env=env, cwd=root, check=True, capture_output=True, text=True)
        timings.append(time.perf_counter() - start)
    result = json.loads(score.read_text(encoding='utf-8'))
    result.update(external_prediction_seconds=timings[0], external_scoring_seconds=timings[1],
                  external_total_seconds=sum(timings), reference_cpu_measurement=False,
                  inference_asset_bytes=before['inference_asset_bytes'], cpu_numeric_threads_limit=4,
                  predictions_sha256=sha(predictions), official_score_sha256=sha(score),
                  timing_includes_startup_imports_loading_preprocessing=True)
    (args.output / 'measurement.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
