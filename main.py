import os
import time
import json
import torch
import argparse
import numpy as np

from engine.logger import Logger
from engine.solver import Trainer
from Data.build_dataloader import build_dataloader, build_dataloader_cond
from Models.interpretable_diffusion.model_utils import unnormalize_to_zero_to_one
from Utils.context_fid import Context_FID
from Utils.cross_correlation import CrossCorrelLoss
from Utils.metric_utils import display_scores
from Utils.io_utils import load_yaml_config, seed_everything, merge_opts_to_config, instantiate_from_config
import os
cpu_num = 2
os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Training Script')
    parser.add_argument('--name', type=str, default=None)

    parser.add_argument('--config_file', type=str, default=None, 
                        help='path of config file')
    parser.add_argument('--output', type=str, default='OUTPUT', 
                        help='directory to save the results')
    parser.add_argument('--tensorboard', action='store_true', 
                        help='use tensorboard for logging')

    # args for random

    parser.add_argument('--cudnn_deterministic', action='store_true', default=False,
                        help='set cudnn.deterministic True')
    parser.add_argument('--seed', type=int, default=12345, 
                        help='seed for initializing training.')  # 12345
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU id to use. If given, only the specific gpu will be'
                        ' used, and ddp will be disabled')
    
    # args for training
    parser.add_argument('--train', action='store_true', default=False, help='Train or Test.')
    parser.add_argument('--sample', type=int, default=0, 
                        choices=[0, 1], help='Condition or Uncondition.')
    parser.add_argument('--mode', type=str, default='infill',
                        help='Infilling or Forecasting.')
    parser.add_argument('--milestone', type=int, default=10)

    parser.add_argument('--missing_ratio', type=float, default=0., help='Ratio of Missing Values.')
    parser.add_argument('--pred_len', type=int, default=0, help='Length of Predictions.')

    # Conditional inference sampling mode (only used when --sample 1 with --mode infill/predict):
    #   ddpm    : original full T-step DDPM sampling (uses sample_infill)
    #   fast200 : DDIM sampling with `--fast_steps` steps (uses fast_sample_infill)
    #   banded  : stridediff frequency-aware dynamic-jump sampling
    #             (uses sample_infill_banded; big_k / med_k / small_k /
    #             last_k_always_micro / tau_energy / tau_dlogP / tau_pv apply).
    #             This is the conditional counterpart of stridediff's
    #             unconditional `sample()` and is expected to be faster
    #             than ddpm while staying close in accuracy, like the
    #             unconditional table.
    parser.add_argument('--inference_mode', type=str, default='ddpm',
                        choices=['ddpm', 'fast200', 'banded', 'custom'],
                        help='Which inference path to use at conditional sampling time.')
    parser.add_argument('--fast_steps', type=int, default=200,
                        help='Number of DDIM steps when --inference_mode=fast200 (default: 200).')

    # Conditional generation task tag. If set, conditional sampling outputs are
    # written to OUTPUT/{name}/cond/{tag}/ as fake_{mode}.npy, real.npy, mask.npy,
    # time_{mode}.json. Used by the conditional experiment pipeline to group
    # (dataset, seed, task) results neatly. If not set, the legacy flat file
    # layout is kept for backward compatibility.
    parser.add_argument('--cond_task_tag', type=str, default=None,
                        help='Optional subfolder tag for conditional outputs.')

    # args for frequency-aware sampling jump sizes
    parser.add_argument('--big_k', type=int, default=30,
                        help='Max jump size when no bands are active.')
    parser.add_argument('--med_k', type=int, default=20,
                        help='Medium jump size when only low bands are active.')
    parser.add_argument('--small_k', type=int, default=1,
                        help='Small jump size when high bands are active.')
    parser.add_argument('--last_k_always_micro', type=int, default=12,
                        help='Use small_k for the last N timesteps.')
    parser.add_argument('--tau_energy', type=float, default=0.5,
                        help='Energy gate threshold for temporal band activity.')
    parser.add_argument('--tau_dlogP', type=float, default=0.01,
                        help='Magnitude drift threshold for temporal band activity.')
    parser.add_argument('--tau_pv', type=float, default=0.08,
                        help='Phase velocity threshold for temporal band activity.')
    # When --inference_mode=banded, run Diffusion-TS langevin_fn after each
    # band-jump to tighten the observed-entry infill loss. Default True to
    # match the original conditional behavior (better MSE). Disable it for
    # pure-jump speed, closer to the unconditional `sample()` regime.
    parser.add_argument('--use_band_langevin', dest='use_band_langevin',
                        action='store_true', default=True,
                        help='[banded] Run langevin_fn per step (default ON).')
    parser.add_argument('--no_band_langevin', dest='use_band_langevin',
                        action='store_false',
                        help='[banded] Skip langevin_fn per step (fast mode).')
    parser.add_argument('--use_band_projection', dest='use_band_projection',
                        action='store_true', default=False,
                        help='[banded] Soft rFFT projection of the update delta (default OFF).')
    parser.add_argument('--no_band_projection', dest='use_band_projection',
                        action='store_false',
                        help='[banded] Disable rFFT projection (default).')
    parser.add_argument('--save_npy', dest='save_npy', action='store_true', default=True,
                        help='Save generated samples to disk (default: True).')
    parser.add_argument('--no_save_npy', dest='save_npy', action='store_false',
                        help='Skip saving generated samples to disk (useful for hparam search).')
    parser.add_argument('--eval_cfid', action='store_true', default=False,
                        help='Compute Context-FID right after sampling.')
    parser.add_argument('--eval_corr', action='store_true', default=False,
                        help='Compute cross-correlation score right after sampling.')
    parser.add_argument('--eval_disc', action='store_true', default=False,
                        help='Compute discriminative score right after sampling (requires TensorFlow).')
    parser.add_argument('--eval_pred', action='store_true', default=False,
                        help='Compute predictive score right after sampling (requires TensorFlow).')
    parser.add_argument('--eval_iterations', type=int, default=5,
                        help='Iterations used for each score group.')
    parser.add_argument('--eval_repeats', type=int, default=3,
                        help='How many score groups to report.')
    
    # args for modify config
    parser.add_argument('opts', help='Modify config options using the command-line',
                        default=None, nargs=argparse.REMAINDER)  

    args = parser.parse_args()
    args.save_dir = os.path.join(args.output, f'{args.name}')

    return args

def main():
    args = parse_args()

    if args.seed is not None:
        seed_everything(args.seed)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
    
    config = load_yaml_config(args.config_file)
    config = merge_opts_to_config(config, args.opts)

    # Translate --inference_mode into dataloader.test_dataset.sampling_steps so
    # engine.solver.Trainer.restore() dispatches to sample_infill (full-T) vs
    # fast_sample_infill (DDIM with `fast_steps`). Only relevant when we're
    # about to run conditional sampling; harmless otherwise.
    if args.sample == 1 and args.mode in ['infill', 'predict']:
        model_params = config['model'].setdefault('params', {})
        train_T = int(model_params.get('timesteps', 1000))
        test_ds = config['dataloader']['test_dataset']
        if args.inference_mode == 'ddpm':
            test_ds['sampling_steps'] = train_T
        elif args.inference_mode == 'fast200':
            assert args.fast_steps <= train_T, \
                f'--fast_steps ({args.fast_steps}) must be <= timesteps ({train_T})'
            test_ds['sampling_steps'] = int(args.fast_steps)
        elif args.inference_mode == 'banded':
            # banded bypasses the sampling_steps-based dispatch inside
            # Trainer.restore; set it to train_T just so any downstream
            # consumer sees a sensible value.
            test_ds['sampling_steps'] = train_T
        # 'custom': leave whatever is in the yaml / passed via opts.

    logger = Logger(args)
    logger.save_config(config)

    model = instantiate_from_config(config['model']).cuda()
    if args.sample == 1 and args.mode in ['infill', 'predict']:
        test_dataloader_info = build_dataloader_cond(config, args)
    dataloader_info = build_dataloader(config, args)
    trainer = Trainer(config=config, args=args, model=model, dataloader=dataloader_info, logger=logger)

    eval_device = f'cuda:{args.gpu}' if args.gpu is not None and torch.cuda.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')

    if args.train:
        trainer.train()
    elif args.sample == 1 and args.mode in ['infill', 'predict']:
        trainer.load(args.milestone)
        dataloader, dataset = test_dataloader_info['dataloader'], test_dataloader_info['dataset']
        coef = config['dataloader']['test_dataset']['coefficient']
        stepsize = config['dataloader']['test_dataset']['step_size']
        sampling_steps = config['dataloader']['test_dataset']['sampling_steps']
        _tic = time.time()
        samples, reals, masks = trainer.restore(
            dataloader, [dataset.window, dataset.var_num], coef, stepsize, sampling_steps,
            banded=(args.inference_mode == 'banded'),
            big_k=args.big_k, med_k=args.med_k, small_k=args.small_k,
            last_k_always_micro=args.last_k_always_micro,
            tau_energy=args.tau_energy, tau_dlogP=args.tau_dlogP, tau_pv=args.tau_pv,
            use_projection=args.use_band_projection,
            use_langevin=args.use_band_langevin,
        )
        elapsed = time.time() - _tic
        if dataset.auto_norm:
            samples = unnormalize_to_zero_to_one(samples)
            reals = unnormalize_to_zero_to_one(reals)
            # samples = dataset.scaler.inverse_transform(samples.reshape(-1, samples.shape[-1])).reshape(samples.shape)
        if args.cond_task_tag:
            cond_dir = os.path.join(args.save_dir, 'cond', args.cond_task_tag)
            os.makedirs(cond_dir, exist_ok=True)
            np.save(os.path.join(cond_dir, f'fake_{args.inference_mode}.npy'), samples)
            # reals and masks depend on (dataset, task) only, not inference_mode.
            real_p = os.path.join(cond_dir, 'real.npy')
            mask_p = os.path.join(cond_dir, 'mask.npy')
            if not os.path.exists(real_p):
                np.save(real_p, reals)
            if not os.path.exists(mask_p):
                np.save(mask_p, masks.astype(np.float32))
            with open(os.path.join(cond_dir, f'time_{args.inference_mode}.json'), 'w') as f:
                json.dump({
                    'mode': args.inference_mode,
                    'task_mode': args.mode,
                    'missing_ratio': float(args.missing_ratio),
                    'pred_len': int(args.pred_len),
                    'sampling_steps': int(sampling_steps),
                    'time_s': float(elapsed),
                    'num_samples': int(samples.shape[0]),
                }, f, indent=2)
            print(f'[Save] {cond_dir} fake_{args.inference_mode}.npy (elapsed={elapsed:.2f}s)')
        else:
            fname = (
                f'ddpm_{args.mode}_{args.name}_{args.inference_mode}'
                f'_big{args.big_k}_med{args.med_k}_small{args.small_k}_last{args.last_k_always_micro}.npy'
            )
            sample_save_path = os.path.join(args.save_dir, fname)
            np.save(sample_save_path, samples)
    else:
        trainer.load(args.milestone)
        dataset = dataloader_info['dataset']
        samples = trainer.sample(
            num=len(dataset),
            size_every=900,
            shape=[dataset.window, dataset.var_num],
            big_k=args.big_k,
            med_k=args.med_k,
            small_k=args.small_k,
            last_k_always_micro=args.last_k_always_micro,
            tau_energy=args.tau_energy,
            tau_dlogP=args.tau_dlogP,
            tau_pv=args.tau_pv,
        )
        if dataset.auto_norm:
            samples = unnormalize_to_zero_to_one(samples)
            # samples = dataset.scaler.inverse_transform(samples.reshape(-1, samples.shape[-1])).reshape(samples.shape)
        if args.save_npy:
            fname = (
                f'ddpm_fake_{args.name}_milestone_{args.milestone}'
                f'_big{args.big_k}_med{args.med_k}_small{args.small_k}_last{args.last_k_always_micro}'
                f'_te{args.tau_energy}_td{args.tau_dlogP}_tp{args.tau_pv}_V2.npy'
            )
            sample_save_path = os.path.join(args.save_dir, fname)
            np.save(sample_save_path, samples)
            print(f'[Save] {sample_save_path}')

        need_eval = args.eval_cfid or args.eval_corr or args.eval_disc or args.eval_pred
        if need_eval:
            if args.eval_iterations <= 0 or args.eval_repeats <= 0:
                raise ValueError('--eval_iterations and --eval_repeats must be positive integers.')

            ori_path = os.path.join(
                args.save_dir,
                'samples',
                f'{args.name}_norm_truth_{dataset.window}_train.npy',
            )
            if not os.path.exists(ori_path):
                raise FileNotFoundError(
                    f'Cannot find normalized training truth file: {ori_path}'
                )

            ori_data = np.load(ori_path)
            fake_data = samples

            print(f'[Eval] real: {ori_path}')
            print(f'[Eval] fake: in-memory samples {fake_data.shape}')
            print(f'[Eval] iterations={args.eval_iterations}, repeats={args.eval_repeats}, device={eval_device}')

        if args.eval_cfid:
            for repeat_idx in range(args.eval_repeats):
                context_fid_score = []
                for iter_idx in range(args.eval_iterations):
                    context_fid = Context_FID(
                        ori_data[:],
                        fake_data[:ori_data.shape[0]],
                        device=eval_device,
                    )
                    context_fid_score.append(context_fid)
                    print(f'[C-FID] Repeat {repeat_idx + 1}, Iter {iter_idx + 1}: {context_fid}')
                print(f'[C-FID] Repeat {repeat_idx + 1} final:')
                display_scores(context_fid_score, tag='C-FID')

        if args.eval_corr:
            x_real = torch.from_numpy(ori_data)
            x_fake = torch.from_numpy(fake_data[:ori_data.shape[0]])
            for repeat_idx in range(args.eval_repeats):
                corr_scores = []
                size = max(int(x_real.shape[0] / args.eval_iterations), 1)
                for iter_idx in range(args.eval_iterations):
                    real_idx = np.random.randint(0, x_real.shape[0], size=(size,))
                    fake_idx = np.random.randint(0, x_fake.shape[0], size=(size,))
                    corr = CrossCorrelLoss(x_real[real_idx, :, :], name='CrossCorrelLoss')
                    loss = corr.compute(x_fake[fake_idx, :, :])
                    corr_scores.append(loss.item())
                    print(f'[Correlation] Repeat {repeat_idx + 1}, Iter {iter_idx + 1}: {loss.item()}')
                print(f'[Correlation] Repeat {repeat_idx + 1} final:')
                display_scores(corr_scores, tag='Correlation')

        if args.eval_disc:
            from Utils.discriminative_metric import discriminative_score_metrics
            for repeat_idx in range(args.eval_repeats):
                disc_scores = []
                for iter_idx in range(args.eval_iterations):
                    temp_disc, fake_acc, real_acc = discriminative_score_metrics(
                        ori_data[:], fake_data[:ori_data.shape[0]]
                    )
                    disc_scores.append(temp_disc)
                    print(f'[Discriminative] Repeat {repeat_idx + 1}, Iter {iter_idx + 1}: {temp_disc}')
                print(f'[Discriminative] Repeat {repeat_idx + 1} final:')
                display_scores(disc_scores, tag='Discriminative')

        if args.eval_pred:
            from Utils.predictive_metric import predictive_score_metrics
            for repeat_idx in range(args.eval_repeats):
                pred_scores = []
                for iter_idx in range(args.eval_iterations):
                    temp_pred = predictive_score_metrics(
                        ori_data, fake_data[:ori_data.shape[0]]
                    )
                    pred_scores.append(temp_pred)
                    print(f'[Predictive] Repeat {repeat_idx + 1}, Iter {iter_idx + 1}: {temp_pred}')
                print(f'[Predictive] Repeat {repeat_idx + 1} final:')
                display_scores(pred_scores, tag='Predictive')

if __name__ == '__main__':
    main()
