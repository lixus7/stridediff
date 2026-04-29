"""Compute MSE for a conditional generation task output (StrideDiff).

Given a folder produced by the conditional sampling step
(see main.py --cond_task_tag), this script loads:

  - real.npy           (real test windows, unnormalized to [0, 1])
  - mask.npy           (1 = observed, 0 = missing/forecast target)
  - fake_{mode}.npy    (generated completion under a given inference mode)
  - time_{mode}.json   (wall-clock sampling time for the mode)

and computes the MSE restricted to the missing cells (mask == 0),
matching the Diffusion-TS paper and Tutorial_2.ipynb.

The result is written to ``metric_{mode}.json`` next to the inputs.

Example:

    python Experiments/compute_cond_metrics.py \
        --cond_dir OUTPUT/stocks_seed12345_L48/cond/infill_mr0.5 \
        --mode fast200
"""
import os
import json
import argparse

import numpy as np
from sklearn.metrics import mean_squared_error


def compute_mse(real: np.ndarray, mask: np.ndarray, fake: np.ndarray) -> float:
    """MSE computed only on the held-out cells (``mask == 0``)."""
    real = np.asarray(real)
    mask = np.asarray(mask).astype(bool)
    fake = np.asarray(fake)
    assert real.shape == mask.shape, (
        f'real shape {real.shape} != mask shape {mask.shape}')
    # fake may have the same shape as real, or add a leading "num_samples" axis.
    if fake.shape == real.shape:
        return float(mean_squared_error(fake[~mask], real[~mask]))
    if fake.ndim == real.ndim + 1 and fake.shape[1:] == real.shape:
        # average over the num_samples dimension before comparing.
        fake = fake.mean(axis=0)
        return float(mean_squared_error(fake[~mask], real[~mask]))
    raise ValueError(
        f'Incompatible shapes: real={real.shape}, mask={mask.shape}, fake={fake.shape}')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cond_dir', type=str, required=True,
                   help='Folder containing real.npy / mask.npy / fake_{mode}.npy.')
    p.add_argument('--mode', type=str, required=True,
                   help='Inference mode label (e.g., ddpm, fast200).')
    p.add_argument('--out', type=str, default=None,
                   help='Output JSON path (default: {cond_dir}/metric_{mode}.json).')
    return p.parse_args()


def main():
    args = parse_args()
    real_p = os.path.join(args.cond_dir, 'real.npy')
    mask_p = os.path.join(args.cond_dir, 'mask.npy')
    fake_p = os.path.join(args.cond_dir, f'fake_{args.mode}.npy')
    time_p = os.path.join(args.cond_dir, f'time_{args.mode}.json')

    for p in (real_p, mask_p, fake_p):
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    real = np.load(real_p)
    mask = np.load(mask_p)
    fake = np.load(fake_p)
    mse = compute_mse(real, mask, fake)

    time_s = None
    if os.path.exists(time_p):
        with open(time_p) as f:
            time_s = float(json.load(f).get('time_s', 0.0))

    out = args.out or os.path.join(args.cond_dir, f'metric_{args.mode}.json')
    result = {
        'mode': args.mode,
        'mse': float(mse),
        'time_s': time_s,
        'num_targets': int((~mask.astype(bool)).sum()),
        'real_shape': list(real.shape),
        'fake_shape': list(fake.shape),
    }
    with open(out, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'[cond-metric] {args.cond_dir} mode={args.mode} mse={mse:.6f} time={time_s}s')


if __name__ == '__main__':
    main()
