# URetinex-Net++

## 统一复现的约定

- 训练输出固定在 `experiments/<实验名>/`，测试输出固定在 `test_result/<实验名>/<数据集名>/`。
- 三个数据集均按配对的相对路径匹配 LQ/GT；训练使用对齐随机裁剪与同一几何增强，验证/测试使用完整图像，不做 resize。
- 默认测试策略是 `official_gt`：**保留作者原始的 GT-derived ratio**。它先从 GT 高照图取得高照 illumination（默认 `net_L=False` 时为 GT RGB 三通道最大值），再计算每张图的全局 ratio：`mean(Q_high / (L_low + 1e-4))`，并限制为不小于 `min_ratio=1`。它没有把 GT 纹理直接输入增强网络，但确实使用了 GT 推导的曝光比例，因此只可作为官方协议复现；不应与仅输入低照图的方法直接作公平部署比较。
- 若需要不依赖 GT 的单图推理，显式改用 `--ratio-mode fixed --ratio <数值>`。`metric.csv` 会记录 `ratio_mode`，防止两种协议混淆。
- 统一指标为逐图计算后算术平均的 RGB PSNR、RGB SSIM 和 LPIPS-Alex v0.1；禁止 resize、GT-Mean 与默认 self-ensemble。复杂度固定用 THOP、`model.eval()`、输入 `1x3x256x256`，同时记录 Params(M)、GMACs 与 GFLOPs。

# 创建环境



```bash
git clone https://github.com/jaxhur/URetinex-Net-PLUS.git
conda create -n uretinex python=3.11 -y
conda activate uretinex
cd URetinex-Net-PLUS
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

python -c "import torch; print('torch=', torch.__version__); print('torch_cuda=', torch.version.cuda); print('cuda_available=', torch.cuda.is_available()); print('gpu=', torch.cuda.get_device_name(0))"
```





## 数据集目录



```
pip install -U gdown
apt install -y unzip

cd ./data
# LOL-v1
gdown "https://drive.google.com/uc?id=1mAN3ll5wWwt1Xz0C7uio31-NJu-50S8Z"
# LOL-v2
gdown "https://drive.google.com/uc?id=1L0UnJg6gZ4Eb7It2EuNxP0L3lQNmKMaP"


# 解压
unzip LOL-v1.zip -d LOL-v1
unzip LOL-v2-renamed.zip -d LOL-v2

rm LOL-v1.zip LOL-v2-renamed.zip
cd ../
```



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



# 训练

`run_reproduction.sh`，会依次完成：`decom_low`、`decom_high`、`unfold`、`adjust`。

- unfolding 会加载两个 decomposition 的最佳权重；
- adjustment 会加载低照 decomposition 与 unfolding 的最佳权重。各阶段的网络、loss、轮数、学习率和默认设置沿用原项目。

| 阶段 | 默认 Epochs | BatchSize | 训练 PatchSize | 验证对象 |
|---|---:|---:|---:|---|
| `decom_low` | 2000 | 4 | `48x48` | 低照重构图 |
| `decom_high` | 300 | 4 | `48x48` | 高照重构图 |
| `unfold` | 2000 | 4 | `48x48` | 高照反射率 |
| `adjust` | 100 | 4 | `96x96` | 最终高照 GT |



```bash
bash run_reproduction.sh --dataset lol-v1 --data-root datasets --experiment uretinex_lolv1 --gpu 0 --parallel-decom --num-workers 4;\
bash run_reproduction.sh --dataset lol-v2-real --data-root datasets --experiment uretinex_lolv2real --gpu 0 --parallel-decom --num-workers 4;\
bash run_reproduction.sh --dataset lol-v2-syn --data-root datasets --experiment uretinex_lolv2syn --gpu 0 --parallel-decom --num-workers 4
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



