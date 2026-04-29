import math
import torch
import torch.nn.functional as F

import os
import numpy as np

import time

from torch import nn
from einops import reduce
from tqdm.auto import tqdm
from functools import partial
from Models.interpretable_diffusion.transformer import Transformer
from Models.interpretable_diffusion.model_utils import default, identity, extract

from typing import List, Sequence, Tuple, Optional, Iterable, Set


# gaussian diffusion trainer class

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class Diffusion_TS(nn.Module):
    def __init__(
            self,
            seq_length,
            feature_size,
            n_layer_enc=3,
            n_layer_dec=6,
            d_model=None,
            timesteps=1000,
            sampling_timesteps=None,
            loss_type='l1',
            beta_schedule='cosine',
            n_heads=4,
            mlp_hidden_times=4,
            eta=0.,
            attn_pd=0.,
            resid_pd=0.,
            kernel_size=None,
            padding_size=None,
            use_ff=True,
            reg_weight=None,
            **kwargs
    ):
        super(Diffusion_TS, self).__init__()

        self.eta, self.use_ff = eta, use_ff
        self.seq_length = seq_length
        self.feature_size = feature_size
        self.ff_weight = default(reg_weight, math.sqrt(self.seq_length) / 5)

        self.model = Transformer(n_feat=feature_size, n_channel=seq_length, n_layer_enc=n_layer_enc, n_layer_dec=n_layer_dec,
                                 n_heads=n_heads, attn_pdrop=attn_pd, resid_pdrop=resid_pd, mlp_hidden_times=mlp_hidden_times,
                                 max_len=seq_length, n_embd=d_model, conv_params=[kernel_size, padding_size], **kwargs)

        self.store_timesteps = False  # A flag to check whether to store the sampling timesteps to compute similarities
        self.save_dir = "" # An aux variable to name the stored output tensors
        self.num_cycles = 0 # An aux variable to store the total number of cycles to be performed
        self.cycle_index = 0 # An aux variable to store the output tensors for the current cycle

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters

        self.sampling_timesteps = default(
            sampling_timesteps, timesteps)  # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.fast_sampling = self.sampling_timesteps < timesteps

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate reweighting
        
        register_buffer('loss_weight', torch.sqrt(alphas) * torch.sqrt(1. - alphas_cumprod) / betas / 100)

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )
    
    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def output(self, x, t, padding_masks=None):
        trend, season = self.model(x, t, padding_masks=padding_masks)
        model_output = trend + season
        return model_output

    def model_predictions(self, x, t, clip_x_start=False, padding_masks=None):
        if padding_masks is None:
            padding_masks = torch.ones(x.shape[0], self.seq_length, dtype=bool, device=x.device)

        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else identity
        x_start = self.output(x, t, padding_masks)
        x_start = maybe_clip(x_start)
        pred_noise = self.predict_noise_from_start(x, t, x_start)
        return pred_noise, x_start

    def p_mean_variance(self, x, t, clip_denoised=True):
        _, x_start = self.model_predictions(x, t)
        if clip_denoised:
            x_start.clamp_(-1., 1.)
        model_mean, posterior_variance, posterior_log_variance = \
            self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start
    
    def p_sample(self, x, t: int, clip_denoised=True, cond_fn=None, model_kwargs=None):
        b, *_, device = *x.shape, self.betas.device
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = \
            self.p_mean_variance(x=x, t=batched_times, clip_denoised=clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        if cond_fn is not None:
            model_mean = self.condition_mean(
                cond_fn, model_mean, model_log_variance, x, t=batched_times, model_kwargs=model_kwargs
            )
        pred_series = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_series, x_start

    # ------------------------------------------------------------
    # Utilities: temporal bands (rFFT bins) and projections
    # ------------------------------------------------------------
    def make_temporal_bands(
            self,
            time_len: int = 24,
            custom: Optional[Sequence[Tuple[int, int]]] = None,
            device: Optional[torch.device] = None,
    ) -> List[torch.Tensor]:
        """
        Build index lists for bands over the 1-D real FFT bins along time.
        rFFT has F = time_len//2 + 1 nonnegative-frequency bins [0..F-1].

        Default coarse bands for time_len=24:
          [0], [1-2], [3-5], [6-12]
        """
        F = time_len // 2 + 1
        if custom is None:
            bands = [(0, 0), (1, 2), (3, 5), (6, F - 1)]
        else:
            bands = custom
        idx: List[torch.Tensor] = []
        for lo, hi in bands:
            lo = max(0, lo)
            hi = min(F - 1, hi)
            if hi < lo:
                hi = lo
            idx.append(torch.arange(lo, hi + 1, device=device))
        return idx

    # ------------------------------------------------------------
    # Band activity: energy ∧ change (log-power drift + phase velocity)
    # ------------------------------------------------------------
    @torch.no_grad()
    def bands_active_temporal(
            self,
            x_prev: torch.Tensor,
            x_curr: torch.Tensor,
            band_indices: Sequence[torch.Tensor],
            tau_energy: float = 0.10,
            tau_dlogP: float = 0.01,
            tau_pv: float = 0.08,
            time_dim: int = -2,
            eps: float = 1e-12,
    ) -> Set[int]:
        """
        Decide which temporal bands are currently 'active'.

        x_prev, x_curr: (..., time, features) at consecutive diffusion indices
        Returns a Python set of active band ids.
        """
        # align dims: time at -2
        if time_dim != x_prev.ndim - 2:
            perm = list(range(x_prev.ndim))
            perm[time_dim], perm[-2] = perm[-2], perm[time_dim]
            x_prev = x_prev.permute(perm)
            x_curr = x_curr.permute(perm)

        # rFFT over time, aggregate power across channels/features
        Xp = torch.fft.rfft(x_prev, dim=-2)  # (..., F, C)
        Xc = torch.fft.rfft(x_curr, dim=-2)  # (..., F, C)
        Pp_f = (Xp.real ** 2 + Xp.imag ** 2).sum(dim=-1)  # (..., F)
        Pc_f = (Xc.real ** 2 + Xc.imag ** 2).sum(dim=-1)  # (..., F)

        # Band power now and at previous step
        P_prev, P_curr = [], []
        for idx in band_indices:
            P_prev.append(Pp_f[..., idx].mean(dim=-1))
            P_curr.append(Pc_f[..., idx].mean(dim=-1))
        P_prev = torch.stack(P_prev, dim=-1)  # (..., B)
        P_curr = torch.stack(P_curr, dim=-1)  # (..., B)

        # Energy gate: fraction of total power
        frac = P_curr / (P_curr.sum(dim=-1, keepdim=True) + eps)  # (..., B)

        # Magnitude drift: |Δ log power|
        dlogP = (P_curr.add(eps).log() - P_prev.add(eps).log()).abs()  # (..., B)

        # Phase velocity (power-weighted; robust when magnitudes small)
        dphi = torch.angle(Xc * torch.conj(Xp))  # (..., F, C)
        pv_b = []
        for idx in band_indices:
            W = Pc_f[..., idx] + eps  # (..., |idx|)
            # average phase over channels first, then power-weight bins
            pv_b.append(torch.sqrt(((W * (dphi[..., idx, :].mean(dim=-1)) ** 2).sum(dim=-1))
                                   / (W.sum(dim=-1) + eps)))
        pv = torch.stack(pv_b, dim=-1)  # (..., B)

        # reduce any leading batch dims by mean
        reduce_dims = tuple(range(frac.ndim - 1))
        frac_m = frac.mean(dim=reduce_dims)
        dlogP_m = dlogP.mean(dim=reduce_dims)
        pv_m = pv.mean(dim=reduce_dims)


        # active_mask = (frac_m >= tau_energy) & ((dlogP_m >= tau_dlogP) | (pv_m >= tau_pv))
        # active_ids = torch.nonzero(active_mask, as_tuple=False).flatten().tolist()
        # return set(int(i) for i in active_ids)

        # --- 核心修改：加强相位卫兵 ---
        # 原逻辑：(能量过关) AND (幅度飘了 OR 相位动了) [cite: 817]
        # 新逻辑：给相位提权。特别是对于高频段，相位动一点就判定为活跃
        active_mask = []
        for i in range(len(band_indices)):
            # 如果是高频带（非 Band 0），我们降低其相位触发的门槛
            # 权重系数 1.5 表示高频相位只要达到阈值的 1/1.5 就触发微采样
            phase_sensitivity = 1.5 if i > 0 else 1.0
            
            is_active = (frac_m[i] >= tau_energy) & \
                        ((dlogP_m[i] >= tau_dlogP) | (pv_m[i] * phase_sensitivity >= tau_pv))
            active_mask.append(is_active)
        
        active_mask = torch.stack(active_mask)
        active_ids = torch.nonzero(active_mask, as_tuple=False).flatten().tolist()
        return set(int(i) for i in active_ids)


    # @torch.no_grad()
    # def project_update_temporal(
    #         self,
    #         delta: torch.Tensor,
    #         band_indices: Sequence[torch.Tensor],
    #         active_idx: Iterable[int],
    #         time_dim: int = -2,
    # ) -> torch.Tensor:
    #     """
    #     Project a time-domain update Δx to (temporal) rFFT bands.

    #     delta: (..., time, features) real
    #     band_indices: list of 1D index tensors over rFFT bins
    #     active_idx: set/list of bands to KEEP (others zeroed)
    #     """
    #     # move time to -2, features to -1
    #     if time_dim != delta.ndim - 2:
    #         perm = list(range(delta.ndim))
    #         perm[time_dim], perm[-2] = perm[-2], perm[time_dim]
    #         delta = delta.permute(perm)

    #     X = torch.fft.rfft(delta, dim=-2)  # (..., F, C)
    #     Fbins = X.shape[-2]
    #     keep = torch.zeros(Fbins, dtype=torch.bool, device=delta.device)
    #     for b in active_idx:
    #         keep[band_indices[b]] = True
    #     X = torch.where(keep[..., None], X, torch.zeros_like(X))
    #     out = torch.fft.irfft(X, n=delta.shape[-2], dim=-2)  # (..., time, C)

    #     # permute back if we permuted in
    #     if time_dim != out.ndim - 2:
    #         inv = list(range(out.ndim))
    #         inv[time_dim], inv[-2] = inv[-2], inv[time_dim]
    #         out = out.permute(inv)
    #     return out

    def project_update_temporal(
                self,
                delta: torch.Tensor,
                band_indices: Sequence[torch.Tensor],
                active_idx: Iterable[int],
                time_dim: int = -2,
                decay_factor: float = 0.2  # <-- 新增：给非活跃带留一点生路
        ) -> torch.Tensor:
            # ... 前面的 permute 逻辑保持不变 ...
            
            X = torch.fft.rfft(delta, dim=-2)
            Fbins = X.shape[-2]
            
            # 创建掩码
            keep_mask = torch.zeros(Fbins, dtype=torch.bool, device=delta.device)
            for b in active_idx:
                keep_mask[band_indices[b]] = True
                
            # --- 核心修改：软门控 ---
            # 活跃频带保留 100% 增量，非活跃频带只保留 decay_factor (如 10%)
            # 这样能防止数值突然断崖式下降导致的“爆炸”
            soft_mask = torch.where(keep_mask[..., None], 
                                    torch.ones_like(X.real), 
                                    torch.full_like(X.real, decay_factor))
            
            X = X * soft_mask
            # -----------------------

            out = torch.fft.irfft(X, n=delta.shape[-2], dim=-2)
            # ... 后面的 permute 逻辑保持不变 ...
            return out    

    # ------------------------------------------------------------
    # DDIM jump (works with a DDPM ε-model) - I adapted DDIM sampler here
    # You can google - DDIM means you do not predict the noise at each step.
    # When the trajectory is not changing much, you re-use the noise predicted from previous step.
    # ------------------------------------------------------------
    @torch.no_grad()
    def ddim_jump(
            self,
            x_t: torch.Tensor,
            t_idx: int,
            t_next: int,
            eps_hat: torch.Tensor,
            x0_pred: torch.Tensor,
            a_bar: torch.Tensor,
            eta: float = 0.0,
    ) -> torch.Tensor:
        """
        Single-eval jump from index t_idx -> t_next (t_next < t_idx).
        a_bar: cumulative alphas with a_bar[0]=1, a_bar[T]≈small; len = T+1
        eps_hat: εθ(x_t, t_idx), same shape as x_t
        """

        # Ensure t_next is not less than -1 before accessing
        safe_t_next = max(t_next, -1)

        #a_t = a_bar[t_idx]
        a_t = a_bar[t_idx + 1]
        # if t_next is skipping several  steps, this will be reflected in alpha_tn
        #a_tn = a_bar[t_next]
        ###a_tn = a_bar[t_next + 1]
        a_tn = a_bar[safe_t_next + 1]

        # predict x0 (disabled, Diffusion-TS is already doing this)
        #x0_pred = (x_t - torch.sqrt(1 - a_t) * eps_hat) / torch.sqrt(a_t)
        x0_pred = x0_pred

        # This term should now be positive because with correct indexing, a_t > a_tn
        term_under_sqrt = (1 - a_tn / a_t) * (1 - a_t) / (1 - a_tn + 1e-12)

        # Add a clamp for extra numerical stability
        sigma = eta * torch.sqrt(torch.clamp(term_under_sqrt, min=0.0))

        # stochasticity for the jump (η=0 deterministic)
        # σ = η * sqrt((1 - a_tn / a_t) * (1 - a_t) / (1 - a_tn))
        # predicting sigma by skipping multiple steps: you see a_tn is for next step
        # thus sigma will be adjusted if you are skipping several steps
        ###############sigma = eta * torch.sqrt((1 - a_tn / a_t) * (1 - a_t) / (1 - a_tn + 1e-12))

        # combine - directly predict x at timestep tn
        dir_xt = torch.sqrt(torch.clamp(1 - a_tn - sigma ** 2, min=0.0)) * eps_hat
        noise = torch.randn_like(x_t) if sigma.item() > 0 else 0.0
        x_tn = torch.sqrt(a_tn) * x0_pred + dir_xt + sigma * noise
        return x_tn

    @torch.no_grad()    
    def dpm_solver_2_jump(self, x_t, t_idx, t_next, eps_curr, eps_prev, a_bar):
            """
            DPM-Solver 二阶多步法跳转。
            利用当前步 t 和前一步 t_old 的 eps 来计算 t_next。
            """
            # 计算 log-SNR (lambda)
            def get_lambda(alpha_bar):
                return 0.5 * torch.log(alpha_bar / (1 - alpha_bar))

            # 这里假设我们拿到的 eps_prev 是距离当前 t 步长约为 k_old 的预测
            # 为了简化，我们直接取 a_bar 对应的位置
            lambda_t = get_lambda(a_bar[t_idx + 1])
            lambda_next = get_lambda(a_bar[t_next + 1])
            
            # 寻找上一个步长的位置（近似处理）
            # 如果你的步长是不固定的，这里需要记录上一个真实的 lambda_prev
            # 简化版：假设上一步的 lambda 是在 lambda_t 之前的某个位置
            h = lambda_next - lambda_t
            h_prev = 0.1 # 这是一个超参数，表示历史预测的权重跨度
            
            r = h_prev / h # 步长比

            # 二阶修正项：利用当前噪声和历史噪声的差值进行外推
            eps_2nd = eps_curr + (1.0 / (2.0 * r)) * (eps_curr - eps_prev)
            
            # 组合成新的 x_next
            phi_1 = torch.expm1(h)
            # 这里的数学推导基于标准 ODE 求解器
            alpha_t = a_bar[t_idx + 1]
            alpha_next = a_bar[t_next + 1]
            
            x_next = (
                torch.sqrt(alpha_next / alpha_t) * x_t 
                - torch.sqrt(1 - alpha_next) * phi_1 * eps_2nd
            )
            return x_next

    @torch.no_grad()
    def sample(self, shape, time_len: int = 24,
               n_bands=None, band_edges: Optional[Sequence[Tuple[int, int]]] = None, low_band_ids=(0,),
               #big_k=4, med_k=2, small_k=1, last_k_always_micro=12,
               big_k=30, med_k=20, small_k=1, last_k_always_micro=12,
               tau_energy=0.5, tau_dlogP=0.01, tau_pv=0.08, device: str = "cuda",
               use_projection=False, eta=0.0, verbose=True, clip_denoised=True):

        # Log the wall-clock time required to sample
        tic = time.time()
        # print("Starting the sampling process...")

        #device = torch.device(device)
        device = self.betas.device

        # Bands
        if band_edges is not None:
            band_idx = self.make_temporal_bands(time_len, custom=band_edges, device=device)
        else:
            band_idx = self.make_temporal_bands(time_len, custom=None, device=device)
        if n_bands is not None:
            # allow overriding #bands by truncation
            band_idx = band_idx[:n_bands]
        BANDS = len(band_idx)
        low_band_ids = set(int(b) for b in low_band_ids if 0 <= int(b) < BANDS)

        # a_bar is self.alphas_cumprod in this file
        a_bar = self.alphas_cumprod
        a_bar = torch.cat([torch.tensor([1.0], device=device), a_bar], dim=0)

        print('self.training (1)', self.training)
        self.training = False
        print('self.training (2)', self.training)

        x = torch.randn(shape, device=device)
        print('Using num_timesteps for t:', self.num_timesteps)
        x_prev_for_gate = x.clone()

        t = self.num_timesteps - 1
        
# --- 新增：初始化历史预测值 ---
        eps_hat_prev = None 
        lambda_prev = None # DPM-Solver 使用对数信噪比空间
        # The while loop is the correct structure for dynamic jumps
        with tqdm(total=self.num_timesteps, desc='sampling loop time step via sample fn') as pbar:
            while t >= 0:
                # Decide active bands using bands_active_temporal
                # Decide active bands based on current state vs previous accepted
                # For example, at the beginning for sines, the active band is Band 0
                # 动态调优：去噪越到后期，相位越敏感
                # 原理：后期信号已成型，微小的相位偏移都会导致波形错位 [cite: 480]
                # 这里我们根据 t 的进度，线性减小 tau_pv（即门槛变低）
                progress = 1.0 - (t / self.num_timesteps)
                current_tau_pv = tau_pv * (1.0 - 0.5 * progress) # 后期灵敏度翻倍
                # 调用活跃带检测，使用更严格的相位阈值
                act = self.bands_active_temporal(
                    x_prev_for_gate, x, band_idx,
                    tau_energy=tau_energy, 
                    tau_dlogP=tau_dlogP, 
                    tau_pv=current_tau_pv, # 使用动态阈值
                    time_dim=-2
                )
                # act = self.bands_active_temporal(
                #     x_prev_for_gate, x, band_idx,
                #     tau_energy=tau_energy, tau_dlogP=tau_dlogP, tau_pv=tau_pv, time_dim=-2
                # )

                #k = 1
                # Choose jump length
                if t <= last_k_always_micro:
                    k = small_k
                elif len(act) == 0:
                    k = min(big_k, t)
                elif set(act).issubset(low_band_ids):
                    k = min(med_k, t)
                else:
                    k = min(small_k, t)
                t_next = t - k

                # Get the model prediction
                t_tensor = torch.full((x.shape[0],), t, device=device, dtype=torch.long)

                # Make a single expensive call to the lowest-level prediction function
                pred_noise, x_start = self.model_predictions(x, t_tensor)

                x0_hat = x_start

                # Perform the clipping that was also present in p_mean_variance
                x0_hat.clamp_(-1., 1.)

                sqrt_alpha_bar_t = self.sqrt_alphas_cumprod[t]
                sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_cumprod[t]

                eps_hat_curr = (x - sqrt_alpha_bar_t * x0_hat) / sqrt_one_minus_alpha_bar_t

                # x_cand = self.ddim_jump(x, t, t_next, eps_hat, x0_hat, a_bar, eta=eta)
# --- 核心逻辑：二阶修正跳转 ---
                if eps_hat_prev is not None and k > 1 and t_next >= 0:
                    # 使用二阶多步法 (DPM-Solver-2 Multistep)
                    x_cand = self.dpm_solver_2_jump(x, t, t_next, eps_hat_curr, eps_hat_prev, a_bar)
                else:
                    # 第一步或微步采样时，退回到标准 DDIM 跳转
                    x_cand = self.ddim_jump(x, t, t_next, eps_hat_curr, x0_hat, a_bar, eta=eta)

                # 更新历史记录
                eps_hat_prev = eps_hat_curr

                #### 修改0318--version1---启用并优化频谱投影（Spectral Projection）
                if not use_projection or len(act) == 0:
                    # 如果不使用投影，或者干脆没有任何活跃频带（全静默状态），直接接受候选值
                    # print('use_projection=False or len(act) == 0')
                    x_new = x_cand
                else:
                    # 计算这一跳带来的增量 delta
                    delta = x_cand - x
                    
                    # 优化点：确保低频基准带（low_band_ids）始终在投影中保留，防止全局趋势丢失
                    # 将当前检测到的活跃带与预设的低频带取并集
                    projection_act = act.union(low_band_ids)
                    
                    # 调用类中定义的 project_update_temporal 进行频域滤波
                    # 它会将 delta 转换到 rFFT 域，只保留 projection_act 中的频段，再逆变换回来
                    filtered_delta = self.project_update_temporal(
                        delta, 
                        band_idx, 
                        active_idx=projection_act, 
                        time_dim=-2
                    )
                    
                    # 更新后的值 = 旧值 + 经过频谱过滤后的增量
                    x_new = x + filtered_delta

                #print('x_cand:', x_cand)

                # Band-projected update (optional - I am actually sure if we need it at all. Play around with it)
                #if not use_projection or len(act) == 0:
                #    x_new = x_cand
                #else:
                #    delta = x_cand - x
                #    x_new = x + self.project_update_temporal(delta, band_idx, active_idx=act, time_dim=-2)

                # Advance
                if verbose:
                    if len(act) == 0:
                        info = "no bands"
                    elif set(act).issubset(low_band_ids):
                        info = f"low bands {sorted(list(act))}"
                    else:
                        info = f"active {sorted(list(act))}"
                    print(f"t={t:4d} -> {t_next:4d} | k={k} | {info}")

                # Update state variables
                x_prev_for_gate = x  # keep last accepted state for gating
                x = x_new
                t = t_next

        # Print the results of the logged the wall-clock time required to sample
        elapsed_time = time.time() - tic
        print(f"\nSampling complete using frequency-aware sampler. Total time taken from Diffusion-TS->sample fn: {elapsed_time:.2f} seconds")

        return x

    @torch.no_grad()
    def fast_sample(self, shape, clip_denoised=True):
        batch, device, total_timesteps, sampling_timesteps, eta = \
            shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.eta

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)

        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        img = torch.randn(shape, device=device)

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step via fast_sample fn'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, clip_x_start=clip_denoised)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            noise = torch.randn_like(img)
            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        return img

    #def generate_mts(self, batch_size=16, model_kwargs=None, cond_fn=None, cache_enabled=False, store_timesteps=False, save_dir=""):
    def generate_mts(
        self,
        batch_size=16,
        model_kwargs=None,
        cond_fn=None,
        cache_enabled=False,
        store_timesteps=False,
        num_cycles=0,
        steps_to_skip=0,
        save_dir="",
        big_k=30,
        med_k=20,
        small_k=1,
        last_k_always_micro=12,
        tau_energy=0.5,
        tau_dlogP=0.01,
        tau_pv=0.08,
    ):
        # Receives the call from the sample method in the solver class

        feature_size, seq_length = self.feature_size, self.seq_length

        print(f"cache_enabled: {cache_enabled}")
        self.model.cache_enabled = cache_enabled
        self.model.decoder.cache_enabled = cache_enabled

        print(f"store_timesteps: {store_timesteps}")
        self.store_timesteps = store_timesteps
        self.model.store_timesteps = store_timesteps

        print(f"num_cycles: {num_cycles}")
        self.num_cycles = num_cycles

        print(f'save_dir: {save_dir}')
        self.save_dir = save_dir

        print(f'steps_to_skip: {steps_to_skip}')
        self.model.force_refresh_limit = steps_to_skip
        self.model.decoder.force_refresh_limit = steps_to_skip

        # the cond_fn code should not run
        if cond_fn is not None:
            sample_fn = self.fast_sample_cond if self.fast_sampling else self.sample_cond
            return sample_fn((batch_size, seq_length, feature_size), model_kwargs=model_kwargs, cond_fn=cond_fn)

        print('self.fast_sampling', self.fast_sampling)
        self.fast_sampling = False  # no idea why this is turning on now ???
        # Resolve jump params (bound method identity differs each access, so branch explicitly)
        _big_k = big_k if big_k is not None else 30
        _med_k = med_k if med_k is not None else 20
        _small_k = small_k if small_k is not None else 1
        _last_micro = last_k_always_micro if last_k_always_micro is not None else 12
        _tau_energy = tau_energy if tau_energy is not None else 0.5
        _tau_dlogP = tau_dlogP if tau_dlogP is not None else 0.01
        _tau_pv = tau_pv if tau_pv is not None else 0.08
        if self.fast_sampling:
            img = self.fast_sample((batch_size, seq_length, feature_size))
        else:
            img = self.sample(
                (batch_size, seq_length, feature_size),
                time_len=seq_length,
                big_k=_big_k,
                med_k=_med_k,
                small_k=_small_k,
                last_k_always_micro=_last_micro,
                tau_energy=_tau_energy,
                tau_dlogP=_tau_dlogP,
                tau_pv=_tau_pv,
            )
        return img

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def _train_loss(self, x_start, t, target=None, noise=None, padding_masks=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        if target is None:
            target = x_start

        x = self.q_sample(x_start=x_start, t=t, noise=noise)  # noise sample
        model_out = self.output(x, t, padding_masks)

        train_loss = self.loss_fn(model_out, target, reduction='none')

        fourier_loss = torch.tensor([0.])
        if self.use_ff:
            fft1 = torch.fft.fft(model_out.transpose(1, 2), norm='forward')
            fft2 = torch.fft.fft(target.transpose(1, 2), norm='forward')
            fft1, fft2 = fft1.transpose(1, 2), fft2.transpose(1, 2)
            fourier_loss = self.loss_fn(torch.real(fft1), torch.real(fft2), reduction='none')\
                           + self.loss_fn(torch.imag(fft1), torch.imag(fft2), reduction='none')
            train_loss +=  self.ff_weight * fourier_loss
        
        train_loss = reduce(train_loss, 'b ... -> b (...)', 'mean')
        train_loss = train_loss * extract(self.loss_weight, t, train_loss.shape)
        return train_loss.mean()

    def forward(self, x, **kwargs):
        b, c, n, device, feature_size, = *x.shape, x.device, self.feature_size
        assert n == feature_size, f'number of variable must be {feature_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self._train_loss(x_start=x, t=t, **kwargs)

    def return_components(self, x, t: int):
        b, c, n, device, feature_size, = *x.shape, x.device, self.feature_size
        assert n == feature_size, f'number of variable must be {feature_size}'
        t = torch.tensor([t])
        t = t.repeat(b).to(device)
        x = self.q_sample(x, t)
        trend, season, residual = self.model(x, t, return_res=True)
        return trend, season, residual, x

    def fast_sample_infill(self, shape, target, sampling_timesteps, partial_mask=None, clip_denoised=True, model_kwargs=None):
        batch, device, total_timesteps, eta = shape[0], self.betas.device, self.num_timesteps, self.eta

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)

        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        img = torch.randn(shape, device=device)

        for time, time_next in tqdm(time_pairs, desc='conditional sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, clip_x_start=clip_denoised)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            pred_mean = x_start * alpha_next.sqrt() + c * pred_noise
            noise = torch.randn_like(img)

            img = pred_mean + sigma * noise
            img = self.langevin_fn(sample=img, mean=pred_mean, sigma=sigma, t=time_cond,
                                   tgt_embs=target, partial_mask=partial_mask, **model_kwargs)
            target_t = self.q_sample(target, t=time_cond)
            img[partial_mask] = target_t[partial_mask]

        img[partial_mask] = target[partial_mask]

        return img

    @torch.no_grad()
    def sample_infill_banded(
        self,
        shape,
        target,
        partial_mask=None,
        model_kwargs=None,
        clip_denoised=True,
        time_len: Optional[int] = None,
        n_bands=None,
        band_edges: Optional[Sequence[Tuple[int, int]]] = None,
        low_band_ids=(0,),
        big_k=30, med_k=20, small_k=1, last_k_always_micro=12,
        tau_energy=0.5, tau_dlogP=0.01, tau_pv=0.08,
        use_projection=False, eta=0.0, verbose=False,
        use_langevin=True,
    ):
        """Conditional (imputation / forecasting) sampling with stridediff's
        dynamic frequency-aware jump schedule.

        Mirrors the unconditional :meth:`sample` (band-activity gating,
        big/med/small jump sizes, last-K-always-micro, DDIM / DPM-Solver-2
        jumps, optional rFFT soft projection) but after each jump:

          * optionally runs the Diffusion-TS ``langevin_fn`` to tighten
            the observed-entry infill loss (same as ``fast_sample_infill``);
          * overlays the observed entries with ``q_sample(target, t_next)``
            so the known region stays consistent with the noise schedule at
            the post-jump step (correct for large k, unlike the Diffusion-TS
            ``fast_sample_infill`` which uses the pre-jump t).
        """
        device = self.betas.device
        if time_len is None:
            time_len = int(shape[-2]) if len(shape) >= 2 else int(self.seq_length)

        if band_edges is not None:
            band_idx = self.make_temporal_bands(time_len, custom=band_edges, device=device)
        else:
            band_idx = self.make_temporal_bands(time_len, custom=None, device=device)
        if n_bands is not None:
            band_idx = band_idx[:n_bands]
        BANDS = len(band_idx)
        low_band_ids_set = set(int(b) for b in low_band_ids if 0 <= int(b) < BANDS)

        a_bar = self.alphas_cumprod
        a_bar = torch.cat([torch.tensor([1.0], device=device), a_bar], dim=0)

        x = torch.randn(shape, device=device)
        x_prev_for_gate = x.clone()
        t = self.num_timesteps - 1
        eps_hat_prev = None

        if model_kwargs is None:
            model_kwargs = {}

        with tqdm(total=self.num_timesteps,
                  desc='cond banded sampling loop time step') as pbar:
            while t >= 0:
                progress = 1.0 - (t / self.num_timesteps)
                current_tau_pv = tau_pv * (1.0 - 0.5 * progress)

                act = self.bands_active_temporal(
                    x_prev_for_gate, x, band_idx,
                    tau_energy=tau_energy,
                    tau_dlogP=tau_dlogP,
                    tau_pv=current_tau_pv,
                    time_dim=-2,
                )

                if t <= last_k_always_micro:
                    k = small_k
                elif len(act) == 0:
                    k = min(big_k, t)
                elif set(act).issubset(low_band_ids_set):
                    k = min(med_k, t)
                else:
                    k = min(small_k, t)
                k = max(int(k), 1)
                t_next = t - k

                t_tensor = torch.full((x.shape[0],), t, device=device, dtype=torch.long)
                _, x_start = self.model_predictions(x, t_tensor, clip_x_start=clip_denoised)
                x0_hat = x_start
                if clip_denoised:
                    x0_hat.clamp_(-1., 1.)

                sqrt_ab_t = self.sqrt_alphas_cumprod[t]
                sqrt_1mab_t = self.sqrt_one_minus_alphas_cumprod[t]
                eps_hat_curr = (x - sqrt_ab_t * x0_hat) / sqrt_1mab_t

                if eps_hat_prev is not None and k > 1 and t_next >= 0:
                    x_cand = self.dpm_solver_2_jump(
                        x, t, t_next, eps_hat_curr, eps_hat_prev, a_bar)
                else:
                    x_cand = self.ddim_jump(
                        x, t, t_next, eps_hat_curr, x0_hat, a_bar, eta=eta)
                eps_hat_prev = eps_hat_curr

                if (not use_projection) or len(act) == 0:
                    x_new = x_cand
                else:
                    delta = x_cand - x
                    projection_act = act.union(low_band_ids_set)
                    filtered_delta = self.project_update_temporal(
                        delta, band_idx, active_idx=projection_act, time_dim=-2,
                    )
                    x_new = x + filtered_delta

                # Diffusion-TS style Langevin refinement on the observed
                # pattern. We match fast_sample_infill's convention of
                # passing t_pre_jump so that self.output inside langevin_fn
                # is evaluated at the (noisy) level the model was trained on.
                if (use_langevin and partial_mask is not None
                        and model_kwargs and 'coef' in model_kwargs
                        and 'learning_rate' in model_kwargs):
                    a_t_scalar = a_bar[t + 1]
                    a_tn_scalar = a_bar[max(t_next, -1) + 1]
                    term = (1 - a_tn_scalar / a_t_scalar) * (1 - a_t_scalar) \
                        / (1 - a_tn_scalar + 1e-12)
                    sigma_scalar = eta * torch.sqrt(torch.clamp(term, min=0.0))
                    sigma = sigma_scalar * torch.ones_like(x_new)
                    x_new = self.langevin_fn(
                        sample=x_new, mean=x_cand, sigma=sigma, t=t_tensor,
                        tgt_embs=target, partial_mask=partial_mask,
                        **model_kwargs,
                    )

                if partial_mask is not None:
                    if t_next >= 0:
                        t_next_tensor = torch.full(
                            (x.shape[0],), t_next,
                            device=device, dtype=torch.long,
                        )
                        target_t = self.q_sample(target, t=t_next_tensor)
                        x_new[partial_mask] = target_t[partial_mask]
                    else:
                        x_new[partial_mask] = target[partial_mask]

                if verbose:
                    if len(act) == 0:
                        info = "no bands"
                    elif set(act).issubset(low_band_ids_set):
                        info = f"low {sorted(list(act))}"
                    else:
                        info = f"active {sorted(list(act))}"
                    print(f"t={t:4d} -> {t_next:4d} | k={k} | {info}")

                pbar.update(int(k))
                x_prev_for_gate = x
                x = x_new
                t = t_next

        if partial_mask is not None:
            x[partial_mask] = target[partial_mask]
        return x

    def sample_infill(
        self,
        shape, 
        target,
        partial_mask=None,
        clip_denoised=True,
        model_kwargs=None,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.
        """
        batch, device = shape[0], self.betas.device
        img = torch.randn(shape, device=device)
        for t in tqdm(reversed(range(0, self.num_timesteps)),
                      desc='conditional sampling loop time step', total=self.num_timesteps):
            img = self.p_sample_infill(x=img, t=t, clip_denoised=clip_denoised, target=target,
                                       partial_mask=partial_mask, model_kwargs=model_kwargs)
        
        img[partial_mask] = target[partial_mask]
        return img
    
    def p_sample_infill(
        self,
        x,
        target,
        t: int,
        partial_mask=None,
        clip_denoised=True,
        model_kwargs=None
    ):
        b, *_, device = *x.shape, self.betas.device
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, _ = \
            self.p_mean_variance(x=x, t=batched_times, clip_denoised=clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        sigma = (0.5 * model_log_variance).exp()
        pred_img = model_mean + sigma * noise

        pred_img = self.langevin_fn(sample=pred_img, mean=model_mean, sigma=sigma, t=batched_times,
                                    tgt_embs=target, partial_mask=partial_mask, **model_kwargs)
        
        target_t = self.q_sample(target, t=batched_times)
        pred_img[partial_mask] = target_t[partial_mask]

        return pred_img

    def langevin_fn(
        self,
        coef,
        partial_mask,
        tgt_embs,
        learning_rate,
        sample,
        mean,
        sigma,
        t,
        coef_=0.
    ):
    
        if t[0].item() < self.num_timesteps * 0.05:
            K = 0
        elif t[0].item() > self.num_timesteps * 0.9:
            K = 3
        elif t[0].item() > self.num_timesteps * 0.75:
            K = 2
            learning_rate = learning_rate * 0.5
        else:
            K = 1
            learning_rate = learning_rate * 0.25

        input_embs_param = torch.nn.Parameter(sample)

        with torch.enable_grad():
            for i in range(K):
                optimizer = torch.optim.Adagrad([input_embs_param], lr=learning_rate)
                optimizer.zero_grad()

                x_start = self.output(x=input_embs_param, t=t)

                if sigma.mean() == 0:
                    logp_term = coef * ((mean - input_embs_param) ** 2 / 1.).mean(dim=0).sum()
                    infill_loss = (x_start[partial_mask] - tgt_embs[partial_mask]) ** 2
                    infill_loss = infill_loss.mean(dim=0).sum()
                else:
                    logp_term = coef * ((mean - input_embs_param)**2 / sigma).mean(dim=0).sum()
                    infill_loss = (x_start[partial_mask] - tgt_embs[partial_mask]) ** 2
                    infill_loss = (infill_loss/sigma.mean()).mean(dim=0).sum()
            
                loss = logp_term + infill_loss
                loss.backward()
                optimizer.step()
                epsilon = torch.randn_like(input_embs_param.data)
                input_embs_param = torch.nn.Parameter((input_embs_param.data + coef_ * sigma.mean().item() * epsilon).detach())

        sample[~partial_mask] = input_embs_param.data[~partial_mask]
        return sample
    
    def condition_mean(self, cond_fn, mean, log_variance, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x=x, t=t, **model_kwargs)
        new_mean = (
            mean.float() + torch.exp(log_variance) * gradient.float()
        )
        return new_mean
    
    def condition_score(self, cond_fn, x_start, x, t, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.

        See condition_mean() for details on cond_fn.

        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = extract(self.alphas_cumprod, t, x.shape)

        eps = self.predict_noise_from_start(x, t, x_start)
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(x, t, **model_kwargs)

        pred_xstart = self.predict_start_from_noise(x, t, eps)
        model_mean, _, _ = self.q_posterior(x_start=pred_xstart, x_t=x, t=t)
        return model_mean, pred_xstart
    
    def sample_cond(
        self,
        shape,
        clip_denoised=True,
        model_kwargs=None,
        cond_fn=None
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.
        """
        batch, device = shape[0], self.betas.device
        img = torch.randn(shape, device=device)
        for t in tqdm(reversed(range(0, self.num_timesteps)),
                      desc='sampling loop time step', total=self.num_timesteps):
            img, x_start = self.p_sample(img, t, clip_denoised=clip_denoised, cond_fn=cond_fn,
                                         model_kwargs=model_kwargs)
        return img

    def fast_sample_cond(
        self,
        shape,
        clip_denoised=True,
        model_kwargs=None,
        cond_fn=None
    ):
        batch, device, total_timesteps, sampling_timesteps, eta = \
            shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.eta

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)

        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        img = torch.randn(shape, device=device)
        x_start = None

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, clip_x_start=clip_denoised)

            if cond_fn is not None:
                _, x_start = self.condition_score(cond_fn, x_start, img, time_cond, model_kwargs=model_kwargs)
                pred_noise = self.predict_noise_from_start(img, time_cond, x_start)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            noise = torch.randn_like(img)
            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        return imgh


if __name__ == '__main__':
    pass
