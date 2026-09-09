# URetinex-Net++

URetinex-Net++ 是论文 *Interpretable Optimization-Inspired Unfolding Network for Low-light Image Enhancement* 的官方 PyTorch 实现。本仓库保留作者原始训练/测试入口，同时新增一套面向 LOL-v1、LOL-v2-syn、LOL-v2-real 的统一复现入口。

## 统一复现的约定

- 训练输出固定在 `experiments/<实验名>/`，测试输出固定在 `test_result/<实验名>/<数据集名>/`。
- 三个数据集均按配对的相对路径匹配 LQ/GT；训练使用对齐随机裁剪与同一几何增强，验证/测试使用完整图像，不做 resize。
- 默认测试策略是 `official_gt`：**保留作者原始的 GT-derived ratio**。它先从 GT 高照图取得高照 illumination（默认 `net_L=False` 时为 GT RGB 三通道最大值），再计算每张图的全局 ratio：`mean(Q_high / (L_low + 1e-4))`，并限制为不小于 `min_ratio=1`。它没有把 GT 纹理直接输入增强网络，但确实使用了 GT 推导的曝光比例，因此只可作为官方协议复现；不应与仅输入低照图的方法直接作公平部署比较。
- 若需要不依赖 GT 的单图推理，显式改用 `--ratio-mode fixed --ratio <数值>`。`metric.csv` 会记录 `ratio_mode`，防止两种协议混淆。
- 统一指标为逐图计算后算术平均的 RGB PSNR、RGB SSIM 和 LPIPS-Alex v0.1；禁止 resize、GT-Mean 与默认 self-ensemble。复杂度固定用 THOP、`model.eval()`、输入 `1x3x256x256`，同时记录 Params(M)、GMACs 与 GFLOPs。

## 40/50 系显卡环境

不要继续使用 README 原先的 PyTorch 1.4 环境。4090 与 5090 建议使用 CUDA 12.8 或更高对应的 PyTorch build；示例使用 CUDA 12.8：

```bash
conda create -n uretinex python=3.11 -y
conda activate uretinex

pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

服务器端先确认实际 runtime，而不是只看 `nvcc`：

```bash
python -c "import torch; print('torch=', torch.__version__); print('torch_cuda=', torch.version.cuda); print('cuda_available=', torch.cuda.is_available()); print('gpu=', torch.cuda.get_device_name(0))"
```

代码和新脚本都不硬编码 GPU 型号、显存或 device index。未传 `--gpu` 时遵从外部 `CUDA_VISIBLE_DEVICES`；传入物理卡号后，脚本会让进程内部只看见一张逻辑 `cuda:0`。每次切换服务器或 4090/5090 后，仍应先在目标卡以临时实验名完成一次真实 batch 的前向/反向冒烟训练，再启动完整实验。VGG 感知损失首次训练时可能需要下载 TorchVision 的 ImageNet 权重；无外网服务器请预先缓存该权重。

## 数据集目录

`--data-root` 可以是任意服务器绝对路径，但其下必须保持以下结构（大小写也要一致）：

```text
<data-root>/
├── LOL-v1/
│   ├── our485/{low,high}/
│   └── eval15/{low,high}/
└── LOL-v2/
    ├── Synthetic/
    │   ├── Train/{Low,Normal}/
    │   └── Test/{Low,Normal}/
    └── Real_captured/
        ├── Train/{Low,Normal}/
        └── Test/{Low,Normal}/
```

LOL-v1 与 LOL-v2-real 原图为 `400x600`（高 x 宽），LOL-v2-syn 原图为 `384x384`。这些是数据集原图尺寸，不等于训练 patch，也不等于复杂度统计的 `256x256` 输入。

## 训练：一个脚本编排四个阶段

新入口是 `run_reproduction.sh`，会依次完成：`decom_low`、`decom_high`、`unfold`、`adjust`。其中 unfolding 会加载两个 decomposition 的最佳权重；adjustment 会加载低照 decomposition 与 unfolding 的最佳权重。各阶段的网络、loss、轮数、学习率和默认设置沿用原项目。

| 阶段 | 默认 Epochs | BatchSize | 训练 PatchSize | 验证对象 |
|---|---:|---:|---:|---|
| `decom_low` | 2000 | 4 | `48x48` | 低照重构图 |
| `decom_high` | 300 | 4 | `48x48` | 高照重构图 |
| `unfold` | 2000 | 4 | `48x48` | 高照反射率 |
| `adjust` | 100 | 4 | `96x96` | 最终高照 GT |

每个数据集实际的 `iterations_per_epoch` 会在数据加载后打印为 `floor(训练集图像数 / 4)`（训练 loader 使用 `drop_last=True`），实际总迭代数是此值乘以该阶段 Epochs。因为当前代码仓库不含三个完整数据集，LOL-v2-syn 与 LOL-v2-real 的最终值必须以服务器上数据就绪后的启动日志为准；不能在未读取数据时伪造总迭代数。

单卡顺序训练 LOL-v1：

```bash
cd /path/to/URetinex-Net-PLUS
bash run_reproduction.sh \
  --dataset lol-v1 \
  --data-root /path/to/datasets \
  --experiment uretinex_lolv1 \
  --gpu 0 \
  --num-workers 4
```

单卡并行时，下面的命令会像两个终端一样让 `decom_low`、`decom_high` 同时共享物理 GPU 0；之后 unfolding 与 adjustment 仍在 GPU 0 顺序执行：

```bash
bash run_reproduction.sh \
  --dataset lol-v1 \
  --data-root /path/to/datasets \
  --experiment uretinex_lolv1 \
  --gpu 0 \
  --parallel-decom \
  --num-workers 4
```

也可把两个 decomposition 分给不同物理卡：

```bash
bash run_reproduction.sh \
  --dataset lol-v1 \
  --data-root /path/to/datasets \
  --experiment uretinex_lolv1 \
  --gpu 0 \
  --parallel-decom \
  --decom-low-gpu 0 \
  --decom-high-gpu 1 \
  --num-workers 4
```

单卡并行不会改变两个 decomposition 的训练目标，等价于在两个终端各启动一个命令；代价是两个独立 PyTorch 进程会共享显存和算力。因此它适合 decomposition 显存占用确实较低的服务器；若出现 OOM 或吞吐明显变差，改回不带 `--parallel-decom` 的顺序执行即可。脚本会等待两个进程都成功完成后，再开始 unfolding。

LOL-v2-syn 与 LOL-v2-real 只需替换数据集名称和实验名：

```bash
bash run_reproduction.sh --dataset lol-v2-syn \
  --data-root /path/to/datasets --experiment uretinex_lolv2syn --gpu 0

bash run_reproduction.sh --dataset lol-v2-real \
  --data-root /path/to/datasets --experiment uretinex_lolv2real --gpu 0
```

训练会自动从 `experiments/<实验名>/training_state/<阶段>_latest.state` 恢复 optimizer、scheduler、epoch、step、global iteration 和历史 best 指标。周期日志每 20 个 global iteration 输出一次，完整测试集验证默认约每 1000 iteration 一次，训练结束也会验证一次。终端输出会原样写入：

统一入口固定 Python/NumPy/PyTorch 的随机种子，并将 `cudnn.benchmark=False`、`cudnn.deterministic=True`；它不强制 `torch.use_deterministic_algorithms`，以免原项目算子在新版 CUDA 上直接报错。因此跨 4090/5090 的结果应以统一指标的稳定范围比较，而不是要求逐位完全相同。

```text
experiments/<实验名>/
├── config/
├── logs/train.log
├── logs/val.log
├── models/
│   ├── decom_low_{latest,best}_G.pth
│   ├── decom_high_{latest,best}_G.pth
│   ├── unfold_{latest,best}_G.pth
│   ├── adjust_{latest,best}_G.pth
│   └── {latest,best}_G.pth             # final adjustment 的标准别名
├── training_state/
└── tb_looger/
```

### 为什么保留 `best_G.pth`，不只用最后一轮

保留 `best_G.pth`，不要只选最后一轮。每阶段同时有 `latest`（最后一次保存、便于精确续训）、按 `--save-freq` 留存的定时权重，以及 `best`：

- 两个 decomposition 的 best 按其各自重构输入图的统一 RGB PSNR 选择；这只是阶段内质量判断。
- unfolding 的 best 按最终反射率相对高照反射率的统一 RGB PSNR 选择。
- adjustment 的 best 按最终增强图相对高照 GT 的统一 RGB PSNR 选择。只有它的最佳权重额外提供 `models/best_G.pth` 标准别名，统一测试默认应传它。

历史最佳 PSNR 与达到该 PSNR 时的 RGB SSIM 会一起保存到 state 和验证日志；因此不会把“当前 SSIM”误写成“最佳 PSNR 对应 SSIM”。

## 最终测试与 `metric.csv`

统一测试只加载 final adjustment 的 `*_G.pth`，保存完整测试集增强图和一行汇总 CSV。默认 `official_gt` 与作者原始测试一致：

```bash
CUDA_VISIBLE_DEVICES=0 python test_reproduction.py \
  --dataset lol-v1 \
  --data-root /path/to/datasets \
  --experiment uretinex_lolv1 \
  --checkpoint experiments/uretinex_lolv1/models/best_G.pth \
  --ratio-mode official_gt \
  --num-workers 4
```

三个数据集的测试命令分别为：

```bash
CUDA_VISIBLE_DEVICES=0 python test_reproduction.py --dataset lol-v1 \
  --data-root /path/to/datasets --experiment uretinex_lolv1 \
  --checkpoint experiments/uretinex_lolv1/models/best_G.pth

CUDA_VISIBLE_DEVICES=0 python test_reproduction.py --dataset lol-v2-syn \
  --data-root /path/to/datasets --experiment uretinex_lolv2syn \
  --checkpoint experiments/uretinex_lolv2syn/models/best_G.pth

CUDA_VISIBLE_DEVICES=0 python test_reproduction.py --dataset lol-v2-real \
  --data-root /path/to/datasets --experiment uretinex_lolv2real \
  --checkpoint experiments/uretinex_lolv2real/models/best_G.pth
```

不使用 GT-derived ratio 的单图协议示例：

```bash
CUDA_VISIBLE_DEVICES=0 python test_reproduction.py \
  --dataset lol-v1 --data-root /path/to/datasets \
  --experiment uretinex_lolv1_fixed5 \
  --checkpoint experiments/uretinex_lolv1/models/best_G.pth \
  --ratio-mode fixed --ratio 5
```

输出目录如下：

```text
test_result/<实验名>/<数据集名>/
├── enhanced/<与低照图一致的相对路径>
├── metric.csv
└── test.log
```

`metric.csv` 记录以下统一口径：

| 字段 | 含义 |
|---|---|
| `psnr` | RGB `[0,255]`、float64、三通道联合 MSE、`crop_border=0` |
| `ssim` | RGB 分通道平均；`11x11` Gaussian，`sigma=1.5`，`crop_border=0` |
| `lpips` | `LPIPS(net='alex', version='0.1')`，RGB `[-1,1]` |
| `params_m` | 所有推理生成网络参数数除以 `1e6` |
| `gmacs_g` | THOP MACs 除以 `1e9`，固定输入 `1x3x256x256` |
| `gflops_g` | `2 * THOP MACs / 1e9` |
| `ratio_mode` | `official_gt` 或 `fixed`，用于区分测试协议 |
| `ratio_value` | `official_gt` 时为 `gt-derived-per-image`，`fixed` 时为实际 ratio |

复杂度统计的 ratio 固定为 1，只是为了让 THOP 接收单输入的固定形状；这不改变实际测试时 `--ratio-mode` 的生成逻辑。部分自定义/非标准算子可能不被 THOP 完整计数，CSV 的 `complexity_note` 会保留这一固定统计说明。

## 作者原始入口（保留）

原始的 `decom_training.py`、`unfolding_training.py`、`adjust_L_training.py`、`test.py` 仍保留。旧命令中的固定 GPU 编号已取消：不传 `--gpu_id` 时遵从 `CUDA_VISIBLE_DEVICES`，传入时仍兼容原参数。原始 `test.py` 的默认 `--ratio None` 仍计算 GT-derived ratio，未被新统一入口替换。

原始论文口径的指标和下面表格仅用于对照，不与本 README 的统一 RGB 指标混用：

| MAE | SSIM | PSNR | LPIPS_COS | LPIPS | DISTS |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0.0589 | 0.8411 | 23.826 | 1.2115 | 0.2311 | 0.1015 |

## Citation

```bibtex
@article{wu2025interpretable,
  title={Interpretable Optimization-Inspired Unfolding Network for Low-Light Image Enhancement},
  author={Wu, Wenhui and Weng, Jian and Zhang, Pingping and Wang, Xu and Yang, Wenhan and Jiang, Jianmin},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2025},
  publisher={IEEE}
}
```

The code is only for non-commercial use. For questions about the original project, contact: wj1997s@163.com.
