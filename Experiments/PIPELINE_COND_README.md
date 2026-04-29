# StrideDiff 条件生成 pipeline（Imputation / Forecasting）

> 对应 `Experiments/run_cond_pipeline.py` + `Experiments/compute_cond_metrics.py` + `Experiments/build_cond_report.py`。
> 只覆盖论文条件生成表里那 4 个数据集：**Stocks / ETTh / Energy / fMRI**。
> 所有命令默认在 `stridediff/` 目录下执行。

---

## 0. 先决条件

- 已安装 `stridediff/requirements.txt` 中的依赖，数据集已放好（`Data/datasets/`）。
- `main.py` 已支持 `--inference_mode {ddpm,fast200}` 和 `--cond_task_tag`（本次改动一起落的）。

## 0.5 和 diffts 条件 pipeline 的关键差异

- **Inference modes**：stridediff 的模型**没有** `force_fast_sample` 开关，所以只支持 `ddpm`（full-T DDPM，走 `sample_infill`）和 `fast200`（DDIM 用 `--fast_steps` 步，走 `fast_sample_infill`）。不支持 `fast`（full-T DDIM）。
- **Unconditional 采样路径**：`generate_mts` 里的 frequency-aware 跳步（`--big_k / --med_k / ...`）**仅在无条件采样时生效**；条件采样走 `sample_infill / fast_sample_infill`，完全不使用这些 band 参数。
- **训练集划分**：条件 pipeline 会自动把 `dataloader.train_dataset.params.proportion` 设成 `0.9`，与 `test_dataset.proportion=0.9` 的 90/10 切分对齐，确保测试窗口未被训练过。
- **默认 `seq_length`**：48（paper §4.5 / Fig. 5 / Fig. 6），与 yaml 里的 24（无条件默认）不同；脚本会同时覆盖 `model.params.seq_length`、`train_dataset.params.window`、`test_dataset.params.window`。
- **独立的 checkpoint 目录**：`./Checkpoints_cond_{ds}_seed{seed}_L{seq}_{seq}/`，不覆盖原有的 `./Checkpoints_{ds}_{seq}/`。

---

## 1. Debug 先过一遍（约 2~3 分钟）

```bash
python Experiments/run_cond_pipeline.py --gpus 0 --debug
```

等价默认：
- `--datasets stocks`
- `--seeds 12345`
- `--modes ddpm,fast200`
- `--tasks infill:0.5`（debug 只跑单任务）
- `--milestone 1`
- `model.params.timesteps=20`、`sampling_timesteps=20`、`save_cycle=50`、`max_epochs=100`
- `fast_steps=10`（fast200 采样只跑 10 步）
- 强制 `--no_skip_existing`
- 所有产物落在 `*_debug` 目录，绝不会和正式跑冲突

Debug 出小报表：

```bash
python Experiments/build_cond_report.py \
  --datasets stocks --seeds 12345 \
  --tasks infill:0.5 \
  --seq_length 48 \
  --md_out Experiments/cond_report_debug.md \
  --tex_out Experiments/cond_report_debug.tex --print
```

---

## 2. 正式跑（默认 seq_length = 48，对齐 paper §4.5 / Fig 5-6）

> `--seq_length` 默认就是 48：脚本会把 `model.params.seq_length` / `train_dataset.params.window` / `test_dataset.params.window` 全部改成 48，并把 ckpt 写到 `Checkpoints_cond_{ds}_seed{seed}_L48_48/`、产物写到 `OUTPUT/{ds}_seed{seed}_L48/cond/{task_tag}/`，与 24 长度的跑完全隔离。

**Imputation（5 个 missing_ratio：10/25/50/75/90%）**

```bash
# 8 卡，4 dataset × 5 seed = 20 个 job，每个 job 顺序跑 train → 5 tasks × 2 modes 的 sample + metric
python Experiments/run_cond_pipeline.py --gpus 0,1,2,3,4,5,6,7
```

**Forecasting（paper Fig. 5 的 pred_len ∈ {24, 36}；想多报就追加 6 / 12）**

```bash
python Experiments/run_cond_pipeline.py --gpus 0,1,2,3,4,5,6,7 \
  --tasks predict:24,predict:36
```

**Imputation + Forecasting 一把跑完**（如果显存 / 时间够）

```bash
python Experiments/run_cond_pipeline.py --gpus 0,1,2,3,4,5,6,7 \
  --tasks infill:0.1,infill:0.25,infill:0.5,infill:0.75,infill:0.9,predict:24,predict:36
```

也可以选数据集 / seed 子集、选 modes：

```bash
# 只跑 etth + energy，两个 seed，两种 mode
python Experiments/run_cond_pipeline.py --gpus 0,1 \
  --datasets etth,energy --seeds 12345,2026 --modes ddpm,fast200
```

**想跑 24 长度的 ablation**：加 `--seq_length 24`（OUTPUT / ckpt 都会换到 `_L24` 目录，不影响 48 的产物）。注意 forecasting 的 `pred_len` 必须 < `seq_length`，所以 24 长度下 `predict:36` 不可用。

### 2.1 复用已有的无条件 ckpt

注意：无条件的 ckpt 是在 `proportion=1.0` 上训的，严格说测试窗口已被模型见过；只建议用来**快速估阶**。此外无条件 ckpt 一般是 `seq_length=24` 训的，所以复用时必须同时指定 `--seq_length 24`。

```bash
python Experiments/run_cond_pipeline.py --gpus 0,1,2,3 \
  --reuse_uncond_ckpt --seq_length 24 --steps sample,metric
```

### 2.2 只重算指标 / 只跑 sample

```bash
# 只跑 sample（metric 以后补）
python Experiments/run_cond_pipeline.py --gpus 0,1,2,3 --steps sample

# 只重算 MSE（已经 sample 好）
python Experiments/run_cond_pipeline.py --gpus 0 --steps metric
```

---

## 3. 文件落点

（下表以默认 `seq_length=48` 为例；跑 `--seq_length 24` 时把 `L48` 改成 `L24` 即可。）

| 类型 | 路径模板 |
| --- | --- |
| 条件训练 ckpt | `Checkpoints_cond_{ds}_seed{seed}_L48_48/checkpoint-{milestone}.pt` |
| 条件产物目录 | `OUTPUT/{ds}_seed{seed}_L48/cond/{task_tag}/` |
| - 真值切片 | `OUTPUT/{ds}_seed{seed}_L48/cond/{task_tag}/real.npy` |
| - 观测掩码 | `OUTPUT/{ds}_seed{seed}_L48/cond/{task_tag}/mask.npy`（`1`=观测，`0`=缺失 / 预测目标） |
| - 生成结果 | `OUTPUT/{ds}_seed{seed}_L48/cond/{task_tag}/fake_{mode}.npy` |
| - 采样耗时 | `OUTPUT/{ds}_seed{seed}_L48/cond/{task_tag}/time_{mode}.json` |
| - 指标 | `OUTPUT/{ds}_seed{seed}_L48/cond/{task_tag}/metric_{mode}.json` |
| 子进程日志 | `Experiments/pipeline_logs/cond__{ds}_seed{seed}_L48__{kind}[_{tag}_{mode}].log` |

`task_tag` 规则：`infill_mr{missing_ratio}`（例：`infill_mr0.5`）或 `predict_pl{pred_len}`（例：`predict_pl24`）。

---

## 4. 出最终报表（Markdown + LaTeX）

```bash
# Imputation（默认 5 个 missing_ratio）
python Experiments/build_cond_report.py \
  --md_out Experiments/cond_report_impute.md \
  --tex_out Experiments/cond_report_impute.tex

# Forecasting
python Experiments/build_cond_report.py \
  --tasks predict:24,predict:36 \
  --md_out Experiments/cond_report_forecast.md \
  --tex_out Experiments/cond_report_forecast.tex
```

> `build_cond_report.py` 也带 `--seq_length`（默认 48），用来定位 `OUTPUT/<ds>_seed<s>_L<seq>/` 目录，务必与 `run_cond_pipeline.py` 用的一致；24 长度的报表 `--seq_length 24` 即可。

表结构：
- 两个指标块：**MSE Score (↓)**、**Time (s) (↓)**。
- 每块 3 行方法：`StrideDiff`（= `ddpm`，full T DDPM）、`StrideDiff-fast`（= `fast200`，200 步 DDIM）、`Ours`（占位，全 0）。
- 列：`Stocks` / `ETTh` / `Energy` / `fMRI`。
- 每格默认仅显示跨 seed 的均值；加 `--show_std` 可切到 mean±std。
- 多任务时：`--layout combined`（默认）把所有 task 合并到一张表里，左侧多一列 `Setting`；`--layout per_task` 每个 task 单独一张表。

---

## 5. 手动单跑（调试用）

```bash
# 单跑一个条件 sample（举例：stocks, seed=12345, ddpm, missing_ratio=0.5, seq_length=48）
python main.py --name stocks_seed12345_L48 --config_file Config/stocks.yaml \
  --gpu 0 --seed 12345 \
  --sample 1 --mode infill --missing_ratio 0.5 \
  --milestone 10 --inference_mode ddpm \
  --cond_task_tag infill_mr0.5 \
  solver.results_folder ./Checkpoints_cond_stock_seed12345_L48 \
  model.params.seq_length 48 \
  dataloader.train_dataset.params.window 48 \
  dataloader.test_dataset.params.window 48

# 单算 MSE
python Experiments/compute_cond_metrics.py \
  --cond_dir OUTPUT/stocks_seed12345_L48/cond/infill_mr0.5 \
  --mode ddpm
```
