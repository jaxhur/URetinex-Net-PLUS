"""按统一指标口径测试 URetinex-Net++ 最终 adjustment 权重。"""

import argparse
import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from metrics.unified_image_metrics import calculate_psnr, calculate_ssim
from network.Math_Module import P, Q
from network.decom import Decom
from repro_runtime import (
    PairedImageDataset,
    create_logger,
    dataset_spec,
    load_checkpoint,
    save_tensor_image,
    tensor_to_rgb255,
)
from utils import define_compositor, define_modelA, define_modelL, define_modelR


class URetinexInference(nn.Module):
    """从最终 adjustment checkpoint 重建完整的单图增强生成路径。"""

    def __init__(self, checkpoint):
        super().__init__()
        if checkpoint.get("stage") != "adjust":
            raise ValueError("统一测试只接受 adjustment 阶段保存的最终 *_G.pth 权重。")

        self.adjust_options = SimpleNamespace(**checkpoint["stage_options"])
        self.unfold_options = SimpleNamespace(**checkpoint["unfold_options"])
        self.model_decom_low = Decom()
        self.model_decom_high = Decom() if self.adjust_options.net_L else None
        self.model_r = define_modelR(self.unfold_options)
        self.model_l = define_modelL(self.unfold_options)
        self.adjust_model = define_modelA(self.adjust_options)
        self.fusion_model = define_compositor(self.adjust_options)
        self.p_solver = P()
        self.q_solver = Q()

        state_dict = checkpoint["state_dict"]
        self.model_decom_low.load_state_dict(state_dict["model_decom"])
        if self.model_decom_high is not None:
            if "model_decom_high" not in state_dict:
                raise KeyError("checkpoint 缺少启用 net_L 所需的 model_decom_high 权重。")
            self.model_decom_high.load_state_dict(state_dict["model_decom_high"])
        self.model_r.load_state_dict(state_dict["model_R"])
        self.model_l.load_state_dict(state_dict["model_L"])
        self.adjust_model.load_state_dict(state_dict["model_A"])
        self.fusion_model.load_state_dict(state_dict["model_compositor"])

    def _unfold(self, low_image):
        """执行作者定义的 Retinex unfolding，并收集参与融合的反射率。"""
        reflectance_results = []
        for step in range(self.unfold_options.round):
            if step == 0:
                reflectance_proxy, illumination_proxy = self.model_decom_low(low_image)
            else:
                gamma = self.unfold_options.gamma + self.unfold_options.Roffset * step
                lamda = self.unfold_options.lamda + self.unfold_options.Loffset * step
                reflectance_proxy = self.p_solver(
                    I=low_image,
                    Q=illumination_proxy,
                    R=reflectance,
                    gamma=gamma,
                )
                illumination_proxy = self.q_solver(
                    I=low_image,
                    P=reflectance_proxy,
                    L=illumination,
                    lamda=lamda,
                )
            reflectance = self.model_r(r=reflectance_proxy, l=illumination_proxy)
            illumination = self.model_l(l=illumination_proxy)
            if step + 1 in self.adjust_options.fusion_layers:
                reflectance_results.append(reflectance)
        if len(reflectance_results) != len(self.adjust_options.fusion_layers):
            raise RuntimeError("fusion_layers 与 unfolding round 不匹配。")
        return reflectance_results, illumination

    def _official_gt_ratio(self, low_illumination, high_image):
        """保留原项目默认的 GT-derived ratio 计算方式。"""
        if high_image is None:
            raise ValueError("official_gt 模式要求仅用于指标的配对高照度 GT。")
        if self.model_decom_high is not None:
            _, high_illumination = self.model_decom_high(high_image)
        else:
            high_illumination, _ = torch.max(high_image, dim=1, keepdim=True)

        ratios = []
        for index in range(low_illumination.size(0)):
            ratio = (
                high_illumination[index : index + 1]
                / (low_illumination[index : index + 1] + 0.0001)
            ).mean()
            ratio = max(float(ratio.item()), float(self.adjust_options.min_ratio))
            ratios.append(torch.full_like(low_illumination[index : index + 1], ratio))
        return torch.cat(ratios, dim=0)

    def generate(self, low_image, ratio_mode="official_gt", high_image=None, ratio=None):
        """按指定 ratio 策略生成增强图，默认忠实保留作者的 GT 引导逻辑。"""
        reflectance_results, illumination = self._unfold(low_image)
        if ratio_mode == "official_gt":
            ratio_map = self._official_gt_ratio(illumination, high_image)
        elif ratio_mode == "fixed":
            if ratio is None:
                raise ValueError("fixed 模式必须通过 --ratio 指定用户增强比例。")
            ratio_map = torch.full_like(illumination, float(ratio))
        else:
            raise ValueError(f"不支持 ratio_mode={ratio_mode!r}。")

        adjusted_illumination = self.adjust_model(l=illumination, alpha=ratio_map)
        enhanced_image, _ = self.fusion_model(reflectance_results, adjusted_illumination)
        return enhanced_image

    def forward(self, low_image):
        """提供固定 ratio=1 的单输入生成路径，供统一 THOP 复杂度统计使用。"""
        return self.generate(low_image, ratio_mode="fixed", ratio=1.0)


def calculate_complexity(model, device):
    """按固定 `1x3x256x256` 和 THOP 口径统计参数量及 MACs/FLOPs。"""
    try:
        from thop import profile
    except ImportError as error:
        raise RuntimeError("缺少 thop；请按 README 安装 requirements.txt。") from error

    model.eval()
    params_m = sum(parameter.numel() for parameter in model.parameters()) / 1e6
    dummy = torch.randn(1, 3, 256, 256, device=device)
    with torch.no_grad():
        macs, _ = profile(model, inputs=(dummy,), verbose=False)
    return {
        "params_m": params_m,
        "gmacs_g": macs / 1e9,
        "gflops_g": 2 * macs / 1e9,
        "input_size": "1x3x256x256",
        "complexity_tool": "thop.profile",
        "complexity_note": "GMACs=THOP返回值/1e9；GFLOPs=2*MACs/1e9；固定 ratio=1 的低照生成路径。",
    }


def _make_lpips(device):
    """构建固定 AlexNet v0.1 和 RGB `[-1,1]` 输入的 LPIPS。"""
    try:
        import lpips
    except ImportError as error:
        raise RuntimeError("缺少 lpips；请按 README 安装 requirements.txt。") from error
    return lpips.LPIPS(net="alex", version="0.1").to(device).eval()


def _per_image_metrics(prediction, target, lpips_model):
    """按固定口径计算一张 RGB 图像的 PSNR、SSIM 与 LPIPS。"""
    pred_rgb255 = tensor_to_rgb255(prediction)
    target_rgb255 = tensor_to_rgb255(target)
    psnr = calculate_psnr(pred_rgb255, target_rgb255, crop_border=0)
    ssim = calculate_ssim(pred_rgb255, target_rgb255, crop_border=0)
    with torch.no_grad():
        lpips_value = lpips_model(
            prediction.unsqueeze(0).clamp(0, 1) * 2 - 1,
            target.unsqueeze(0).clamp(0, 1) * 2 - 1,
        ).item()
    return float(psnr), float(ssim), float(lpips_value)


def run_test(args):
    """保存完整测试集增强图，并写入一行可追溯的 metric.csv。"""
    if not torch.cuda.is_available():
        raise RuntimeError("此测试入口要求 CUDA GPU；请在远程 4090/5090 服务器运行。")
    if args.ratio_mode == "fixed" and args.ratio is None:
        raise ValueError("--ratio-mode fixed 时必须额外传入 --ratio。")

    device = torch.device("cuda")
    checkpoint_path = Path(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    model = URetinexInference(checkpoint).to(device).eval()
    complexity = calculate_complexity(model, device)
    lpips_model = _make_lpips(device)

    spec = dataset_spec(args.data_root, args.dataset)
    test_dataset = PairedImageDataset(spec["test_low"], spec["test_high"], training=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    result_dir = Path("test_result") / args.experiment / spec["name"]
    enhanced_dir = result_dir / "enhanced"
    result_dir.mkdir(parents=True, exist_ok=True)
    logger = create_logger(
        f"uretinex.{args.experiment}.{args.dataset}.test", result_dir / "test.log"
    )
    logger.info(
        f"[{args.experiment}][TEST] [dataset: {spec['name']}] "
        f"[ratio_mode: {args.ratio_mode}] [checkpoint: {checkpoint_path}]"
    )
    logger.info(
        f"[{args.experiment}][COMPLEXITY] "
        f"[Params(M): {complexity['params_m']:.6f}] "
        f"[GMACs(G): {complexity['gmacs_g']:.6f}] "
        f"[GFLOPs(G): {complexity['gflops_g']:.6f}] "
        f"[input: {complexity['input_size']}]"
    )

    psnr_values = []
    ssim_values = []
    lpips_values = []
    with torch.no_grad():
        for batch in test_loader:
            low_image = batch["low_light_img"].to(device, non_blocking=True)
            high_image = batch["high_light_img"].to(device, non_blocking=True)
            prediction = model.generate(
                low_image,
                ratio_mode=args.ratio_mode,
                high_image=high_image if args.ratio_mode == "official_gt" else None,
                ratio=args.ratio,
            ).clamp(0, 1)
            psnr, ssim, lpips_value = _per_image_metrics(
                prediction[0], high_image[0], lpips_model
            )
            psnr_values.append(psnr)
            ssim_values.append(ssim)
            lpips_values.append(lpips_value)
            relative_path = batch["relative_path"][0]
            save_tensor_image(prediction[0], enhanced_dir / relative_path)

    row = {
        "experiment": args.experiment,
        "dataset": spec["name"],
        "train_split": spec["train_split"],
        "test_split": spec["test_split"],
        "psnr": f"{np.mean(psnr_values):.6f}",
        "psnr_mode": "BasicSR-RGB-crop0",
        "ssim": f"{np.mean(ssim_values):.6f}",
        "ssim_mode": "BasicSR-RGB-channel-mean-crop0",
        "lpips": f"{np.mean(lpips_values):.6f}",
        "lpips_backbone": "alex",
        "lpips_version": "0.1",
        "lpips_range": "[-1,1]",
        "params_m": f"{complexity['params_m']:.6f}",
        "gmacs_g": f"{complexity['gmacs_g']:.6f}",
        "gflops_g": f"{complexity['gflops_g']:.6f}",
        "input_size": complexity["input_size"],
        "checkpoint": str(checkpoint_path),
        "enhanced_images": str(enhanced_dir),
        "ratio_mode": args.ratio_mode,
        "ratio_value": (
            "gt-derived-per-image"
            if args.ratio_mode == "official_gt"
            else f"{args.ratio:.6f}"
        ),
        "metric_source": "unified_reproduction",
        "complexity_tool": complexity["complexity_tool"],
        "complexity_note": complexity["complexity_note"],
        "resize": "false",
        "gt_mean": "false",
        "self_ensemble": "false",
    }
    csv_path = result_dir / "metric.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    logger.info(
        f"[{args.experiment}][TEST] [images={len(test_dataset)}] "
        f"[psnr={row['psnr']}, rgb_ssim={row['ssim']}, lpips_alex={row['lpips']}] "
        f"[metric_csv={csv_path}]"
    )


def build_parser():
    """创建统一测试命令行参数。"""
    parser = argparse.ArgumentParser(description="URetinex-Net++ 统一三数据集测试")
    parser.add_argument("--dataset", choices=["lol-v1", "lol-v2-syn", "lol-v2-real"], required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ratio-mode", choices=["official_gt", "fixed"], default="official_gt")
    parser.add_argument("--ratio", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser


if __name__ == "__main__":
    run_test(build_parser().parse_args())
