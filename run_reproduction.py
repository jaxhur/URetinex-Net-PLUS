"""按 LOL-v1、LOL-v2-syn、LOL-v2-real 执行 URetinex-Net++ 单阶段训练。"""

import argparse
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from metrics.unified_image_metrics import calculate_psnr, calculate_ssim
from model import AdjustModel, DecomModel, UnfoldingModel
from repro_runtime import (
    LOL_LAYOUTS,
    PairedImageDataset,
    create_loader,
    create_logger,
    dataset_spec,
    format_duration,
    load_checkpoint,
    save_json,
    set_random_seed,
    tensor_to_rgb255,
)


STAGE_DEFAULTS = {
    "decom_low": {"epochs": 2000, "batch_size": 4, "patch_size": 48},
    "decom_high": {"epochs": 300, "batch_size": 4, "patch_size": 48},
    "unfold": {"epochs": 2000, "batch_size": 4, "patch_size": 48},
    "adjust": {"epochs": 100, "batch_size": 4, "patch_size": 96},
}


def _as_float(value):
    """将标量 Tensor 或 Python 数值转换为日志可写的 float。"""
    if isinstance(value, torch.Tensor):
        return float(value.detach().mean().cpu().item())
    return float(value)


def _to_device(batch, device):
    """仅移动模型所需的图像 Tensor，保留相对路径元数据。"""
    return {
        "low_light_img": batch["low_light_img"].to(device, non_blocking=True),
        "high_light_img": batch["high_light_img"].to(device, non_blocking=True),
        "relative_path": batch["relative_path"],
    }


def _build_stage_options(stage, args, spec, experiment_dir):
    """构建并固定原项目各阶段的网络与损失超参数。"""
    common = {
        "batch_size": args.batch_size,
        "size": args.patch_size,
        "n_cpu": args.num_workers,
        "epoch": args.epochs,
        "iteration_to_print": args.print_freq,
        "patch_low": str(spec["train_low"]),
        "patch_high": str(spec["train_high"]),
        "eval_low": str(spec["test_low"]),
        "eval_high": str(spec["test_high"]),
        "saving_eval_dir": str(experiment_dir / "validation" / stage),
    }
    model_dir = experiment_dir / "models"
    if stage == "decom_low":
        return SimpleNamespace(
            **common,
            img_light="low",
            eval_epoch=1,
            init="normal",
            decom_model_dir=str(model_dir),
            model_path=str(model_dir / "decom_low_best_G.pth"),
        )
    if stage == "decom_high":
        return SimpleNamespace(
            **common,
            img_light="high",
            eval_epoch=1,
            init="normal",
            decom_model_dir=str(model_dir),
            model_path=str(model_dir / "decom_high_best_G.pth"),
        )
    if stage == "unfold":
        return SimpleNamespace(
            **common,
            lr=1e-4,
            round=3,
            gamma=0.5,
            lamda=0.5,
            Roffset=0.05,
            Loffset=0.05,
            R_model="HalfDnCNNSE",
            L_model="Illumination_Alone",
            init="normal",
            milestones=[30000],
            loss_options="RL2(1.0)-RSSIM(1.0)-RVGG(1.0)-Ltv(20)",
            l_Pconstraint=1.0,
            l_Qconstraint=1.0,
            l_Ltv=20.0,
            l_R_l2=1.0,
            l_R_ssim=1.0,
            l_R_vgg=1.0,
            freeze_decom=True,
            second_stage="False",
            concat_L=True,
            Decom_model_low_path=str(model_dir / "decom_low_best_G.pth"),
            Decom_model_high_path=str(model_dir / "decom_high_best_G.pth"),
            pretrain_unfolding_model_path="",
            unfolding_model_dir=str(model_dir),
            log_dir=str(experiment_dir / "tb_looger" / "unfold"),
            write_imgs=0,
        )
    if stage == "adjust":
        return SimpleNamespace(
            **common,
            fusion_model="weight3",
            A_model="naive",
            fusion_layers=[1, 2, 3],
            net_L=False,
            milestones=[12300, 25000, 60000],
            min_ratio=1.0,
            l_grad=1.0,
            l_spa=10.0,
            adjust_L_loss="rec-grad-spatial",
            init="xavier",
            adjust_model_dir=str(model_dir),
            Decom_model_low_path=str(model_dir / "decom_low_best_G.pth"),
            Decom_model_high_path=str(model_dir / "decom_high_best_G.pth"),
            pretrain_unfolding_model_path=str(model_dir / "unfold_best_G.pth"),
        )
    raise ValueError(f"未知 stage：{stage}")


def _stage_model(stage, options, device):
    """实例化一个阶段模型，并移动到当前可见的单张 GPU。"""
    if stage.startswith("decom"):
        return DecomModel(options).to(device)
    if stage == "unfold":
        return UnfoldingModel(options).to(device)
    if stage == "adjust":
        return AdjustModel(options).to(device)
    raise ValueError(f"未知 stage：{stage}")


def _model_state(stage, model):
    """按原项目兼容键名导出每个阶段的生成网络权重。"""
    if stage.startswith("decom"):
        return {"model_R": model.decomModel.state_dict()}
    if stage == "unfold":
        return {
            "model_R": model.model_R.state_dict(),
            "model_L": model.model_L.state_dict(),
        }
    if stage == "adjust":
        state_dict = {
            "model_decom": model.model_Decom_low.state_dict(),
            "model_R": model.model_R.state_dict(),
            "model_L": model.model_L.state_dict(),
            "model_A": model.adjust_model.state_dict(),
            "model_compositor": model.fusion_model.state_dict(),
        }
        if model.model_Decom_high is not None:
            state_dict["model_decom_high"] = model.model_Decom_high.state_dict()
        return state_dict
    raise ValueError(f"未知 stage：{stage}")


def _load_model_state(stage, model, state_dict):
    """恢复对应阶段的生成网络状态。"""
    if stage.startswith("decom"):
        model.decomModel.load_state_dict(state_dict["model_R"])
    elif stage == "unfold":
        model.model_R.load_state_dict(state_dict["model_R"])
        model.model_L.load_state_dict(state_dict["model_L"])
    elif stage == "adjust":
        model.model_Decom_low.load_state_dict(state_dict["model_decom"])
        if model.model_Decom_high is not None and "model_decom_high" in state_dict:
            model.model_Decom_high.load_state_dict(state_dict["model_decom_high"])
        model.model_R.load_state_dict(state_dict["model_R"])
        model.model_L.load_state_dict(state_dict["model_L"])
        model.adjust_model.load_state_dict(state_dict["model_A"])
        model.fusion_model.load_state_dict(state_dict["model_compositor"])
    else:
        raise ValueError(f"未知 stage：{stage}")


def _optimizer_and_scheduler(stage, model):
    """获取每个阶段实际参与优化的 optimizer 和 scheduler。"""
    if stage.startswith("decom"):
        return model.optimizer_D, None
    if stage == "unfold":
        return model.optimizer_G, model.scheduler
    if stage == "adjust":
        return model.optimizer_A, model.scheduler
    raise ValueError(f"未知 stage：{stage}")


def _run_unfolding_inference(model, low_image, high_image):
    """执行原 unfolding 公式，并返回最终反射率与高照反射率监督。"""
    for step in range(model.opts.round):
        if step == 0:
            predicted_reflectance, low_illumination = model.model_Decom_low(low_image)
            target_reflectance, _ = model.model_Decom_high(high_image)
        else:
            gamma = model.opts.gamma + model.opts.Roffset * step
            lamda = model.opts.lamda + model.opts.Loffset * step
            predicted_reflectance = model.P(
                I=low_image,
                Q=low_illumination,
                R=restored_reflectance,
                gamma=gamma,
            )
            low_illumination = model.Q(
                I=low_image,
                P=predicted_reflectance,
                L=enhanced_illumination,
                lamda=lamda,
            )
        restored_reflectance = model.model_R(
            r=predicted_reflectance, l=low_illumination
        )
        enhanced_illumination = model.model_L(l=low_illumination)
    return restored_reflectance, target_reflectance


def _run_adjustment_inference(model, low_image, high_image):
    """保留作者默认的 GT-derived ratio，生成最终增强图。"""
    reflectance_list, low_illumination = model.unfolding_inference(low_image)
    high_illumination = model.make_high_L(high_image)
    ratio = model.get_ratio(high_l=high_illumination, low_l=low_illumination)
    adjusted_illumination = model.adjust_model(l=low_illumination, alpha=ratio)
    enhanced_image, _ = model.fusion_model(reflectance_list, adjusted_illumination)
    return enhanced_image


def _mean_metrics(predictions, targets):
    """按逐图算术平均的统一 RGB PSNR/SSIM 口径汇总验证指标。"""
    psnr_values = []
    ssim_values = []
    for prediction, target in zip(predictions, targets):
        pred_rgb255 = tensor_to_rgb255(prediction)
        target_rgb255 = tensor_to_rgb255(target)
        psnr_values.append(calculate_psnr(pred_rgb255, target_rgb255, crop_border=0))
        ssim_values.append(calculate_ssim(pred_rgb255, target_rgb255, crop_border=0))
    return float(np.mean(psnr_values)), float(np.mean(ssim_values))


def _validate(stage, model, loader, device):
    """在完整测试集上验证当前阶段，并恢复模型训练状态。"""
    model.eval()
    psnr_values = []
    ssim_values = []
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            low_image = batch["low_light_img"]
            high_image = batch["high_light_img"]
            if stage.startswith("decom"):
                source = low_image if stage == "decom_low" else high_image
                # 同一次分解前向同时得到 R 和 L，避免验证时重复计算。
                reflectance, illumination = model.decomModel(source)
                prediction = (reflectance * illumination).clamp(0, 1)
                target = source
            elif stage == "unfold":
                prediction, target = _run_unfolding_inference(model, low_image, high_image)
                prediction = prediction.clamp(0, 1)
                target = target.clamp(0, 1)
            else:
                prediction = _run_adjustment_inference(model, low_image, high_image).clamp(0, 1)
                target = high_image.clamp(0, 1)

            batch_psnr, batch_ssim = _mean_metrics(prediction, target)
            psnr_values.append(batch_psnr)
            ssim_values.append(batch_ssim)
    model.train()
    return float(np.mean(psnr_values)), float(np.mean(ssim_values))


def _train_one_batch(stage, model, batch):
    """沿用原项目的三阶段前向、loss 和 optimizer 更新逻辑。"""
    if stage.startswith("decom"):
        return model(batch), model.optimizer_D.param_groups[0]["lr"]
    if stage == "unfold":
        _, losses, learning_rate = model(batch)
        return losses, learning_rate
    if stage == "adjust":
        losses, learning_rate = model(batch)
        return losses, learning_rate
    raise ValueError(f"未知 stage：{stage}")


def _weighted_loss_values(stage, options, losses):
    """将原模型返回的原始 loss 转为与 total_loss 可核对的加权日志项。"""
    values = {"total_loss": losses["total_loss"]}
    if stage.startswith("decom"):
        values["rec_loss"] = losses["rec_loss"]
        values["L_supervised_w"] = 0.1 * losses["L_supervised"]
        if "L_aware" in losses:
            values["L_aware_w"] = 0.1 * losses["L_aware"]
        return values

    if stage == "unfold":
        weights = {
            "P_loss": ("P_constraint_w", options.l_Pconstraint),
            "Q_loss": ("Q_constraint_w", options.l_Qconstraint),
            "R_L2_loss": ("R_L2_w", options.l_R_l2),
            "L_tv_loss": ("L_tv_w", options.l_Ltv),
            "R_ssim_loss": ("R_ssim_w", options.l_R_ssim),
            "R_vgg_loss": ("R_vgg_w", options.l_R_vgg),
        }
    elif stage == "adjust":
        weights = {
            "grad": ("grad_w", options.l_grad),
            "high_rec": ("high_rec", 1.0),
            "spatial": ("spatial_w", options.l_spa),
        }
    else:
        raise ValueError(f"未知 stage：{stage}")

    for source_name, (log_name, weight) in weights.items():
        if source_name in losses:
            values[log_name] = float(weight) * losses[source_name]
    return values


def _checkpoint_paths(experiment_dir, stage):
    """返回统一 experiments 目录下的模型和状态文件位置。"""
    model_dir = experiment_dir / "models"
    state_dir = experiment_dir / "training_state"
    model_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    return {
        "latest_model": model_dir / f"{stage}_latest_G.pth",
        "best_model": model_dir / f"{stage}_best_G.pth",
        "latest_state": state_dir / f"{stage}_latest.state",
        "model_dir": model_dir,
        "state_dir": state_dir,
    }


def _save_checkpoint(experiment_dir, stage, payload, global_iter, is_best, periodic):
    """保存 latest、best、定时权重和包含优化器的 training state。"""
    paths = _checkpoint_paths(experiment_dir, stage)
    torch.save(payload, paths["latest_model"])
    torch.save(payload, paths["latest_state"])
    if periodic:
        torch.save(payload, paths["model_dir"] / f"{stage}_{global_iter:08d}_G.pth")
        torch.save(payload, paths["state_dir"] / f"{stage}_{global_iter:08d}.state")
    if is_best:
        torch.save(payload, paths["best_model"])
    if stage == "adjust":
        # 最终阶段提供固定别名，便于测试命令只传一个标准权重路径。
        shutil.copy2(paths["latest_model"], paths["model_dir"] / "latest_G.pth")
        if is_best:
            shutil.copy2(paths["best_model"], paths["model_dir"] / "best_G.pth")


def _resume_state(experiment_dir, stage):
    """自动读取一个阶段最新的完整训练 state。"""
    state_path = _checkpoint_paths(experiment_dir, stage)["latest_state"]
    return load_checkpoint(state_path) if state_path.exists() else None


def _build_payload(stage, model, options, epoch, step, global_iter, best_psnr, best_ssim):
    """构建能精确恢复优化器、调度器与历史最佳值的断点。"""
    optimizer, scheduler = _optimizer_and_scheduler(stage, model)
    payload = {
        "stage": stage,
        "epoch": epoch,
        "step": step,
        "global_iter": global_iter,
        "state_dict": _model_state(stage, model),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "opts": options,
        "stage_options": vars(options),
        "best_psnr": best_psnr,
        "best_rgb_ssim": best_ssim,
    }
    if stage == "adjust":
        payload["unfold_options"] = vars(model.unfolding_model_opts)
    return payload


def _format_train_message(args, epoch, step, steps_per_epoch, global_iter, total_iter, elapsed, learning_rate, losses):
    """生成固定字段顺序的每 20 iter 训练状态行。"""
    total_loss = losses.get("total_loss", sum(losses.values()))
    loss_text = ", ".join(
        f"{name}={value:.6f}"
        for name, value in sorted(losses.items())
        if name != "total_loss"
    )
    eta = elapsed / max(global_iter, 1) * max(total_iter - global_iter, 0)
    return (
        f"[{args.experiment}][TRAIN] [stage: {args.stage}] "
        f"[progress: epoch={epoch + 1}/{args.epochs}, "
        f"iter={global_iter:,}/{total_iter:,}, step={step}/{steps_per_epoch}] "
        f"[time: elapsed={format_duration(elapsed)}, eta={format_duration(eta)}] "
        f"[optim: lr={learning_rate:.3e}] [total_loss: {total_loss:.6f}] "
        f"[loss: {loss_text}]"
    )


def _format_val_message(args, stage, epoch, global_iter, psnr, ssim, best_psnr, best_ssim, improved):
    """生成固定字段顺序的完整测试集验证状态行。"""
    status = "updated" if improved else "kept"
    best_ssim_text = "unknown" if best_ssim is None else f"{best_ssim:.6f}"
    return (
        f"[{args.experiment}][VAL] [stage: {stage}] "
        f"[progress: epoch={epoch + 1}/{args.epochs}, iter={global_iter:,}] "
        f"[metric: psnr={psnr:.6f}, rgb_ssim={ssim:.6f}, "
        f"ssim_mode=BasicSR-RGB-channel-mean-crop0] "
        f"[best: psnr={best_psnr:.6f}, rgb_ssim={best_ssim_text}, status={status}]"
    )


def train_stage(args):
    """训练一个阶段，并在约每 1000 iter 做完整测试集验证。"""
    if not torch.cuda.is_available():
        raise RuntimeError("此复现入口要求 CUDA GPU；请在远程 4090/5090 服务器运行。")

    device = torch.device("cuda")
    set_random_seed(args.seed)
    spec = dataset_spec(args.data_root, args.dataset)
    experiment_dir = Path("experiments") / args.experiment
    (experiment_dir / "logs").mkdir(parents=True, exist_ok=True)
    train_logger = create_logger(
        f"uretinex.{args.experiment}.{args.stage}.train",
        experiment_dir / "logs" / "train.log",
    )
    val_logger = create_logger(
        f"uretinex.{args.experiment}.{args.stage}.val",
        experiment_dir / "logs" / "val.log",
    )

    options = _build_stage_options(args.stage, args, spec, experiment_dir)
    save_json(vars(options), experiment_dir / "config" / f"{args.stage}.json")
    train_dataset = PairedImageDataset(
        spec["train_low"],
        spec["train_high"],
        patch_size=args.patch_size,
        training=True,
        seed=args.seed,
    )
    val_dataset = PairedImageDataset(spec["test_low"], spec["test_high"], training=False)
    val_loader = create_loader(
        val_dataset, batch_size=1, training=False, num_workers=args.num_workers, seed=args.seed
    )

    model = _stage_model(args.stage, options, device)
    optimizer, scheduler = _optimizer_and_scheduler(args.stage, model)
    state = _resume_state(experiment_dir, args.stage)
    start_epoch = 0
    resume_step = 0
    global_iter = 0
    best_psnr = float("-inf")
    best_ssim = None
    if state is not None:
        _load_model_state(args.stage, model, state["state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None and state.get("scheduler") is not None:
            scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state["epoch"])
        resume_step = int(state["step"])
        global_iter = int(state["global_iter"])
        best_psnr = float(state.get("best_psnr", float("-inf")))
        best_ssim = state.get("best_rgb_ssim")
        train_logger.info(
            f"[{args.experiment}][RESUME] [stage: {args.stage}] "
            f"[epoch={start_epoch + 1}, step={resume_step}, iter={global_iter:,}]"
        )

    train_logger.info(
        f"[{args.experiment}][START] [stage: {args.stage}] "
        f"[dataset: {spec['name']}] [device: {torch.cuda.get_device_name(0)}] "
        f"[torch: {torch.__version__}] [torch_cuda: {torch.version.cuda}]"
    )
    writer_path = experiment_dir / "tb_looger" / args.stage
    writer_path.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(writer_path))
    started_at = time.monotonic()
    last_val_iter = -1
    for epoch in range(start_epoch, args.epochs):
        train_dataset.set_epoch(epoch)
        train_loader = create_loader(
            train_dataset,
            batch_size=args.batch_size,
            training=True,
            num_workers=args.num_workers,
            seed=args.seed + epoch,
        )
        steps_per_epoch = len(train_loader)
        if steps_per_epoch == 0:
            raise RuntimeError(
                f"训练集 {len(train_dataset)} 张，小于 BatchSize={args.batch_size} 且 drop_last=True。"
            )
        total_iter = steps_per_epoch * args.epochs
        train_logger.info(
            f"Training the {epoch + 1} epoch ... [stage={args.stage}, "
            f"iterations_per_epoch={steps_per_epoch}, total_iterations={total_iter}]"
        )
        for step, batch in enumerate(train_loader, start=1):
            if epoch == start_epoch and step <= resume_step:
                continue
            batch = _to_device(batch, device)
            losses, learning_rate = _train_one_batch(args.stage, model, batch)
            raw_loss_values = {
                name: _as_float(value) for name, value in losses.items()
            }
            loss_values = _weighted_loss_values(
                args.stage, options, raw_loss_values
            )
            global_iter += 1
            elapsed = time.monotonic() - started_at
            if global_iter % args.print_freq == 0:
                train_logger.info(
                    _format_train_message(
                        args,
                        epoch,
                        step,
                        steps_per_epoch,
                        global_iter,
                        total_iter,
                        elapsed,
                        learning_rate,
                        loss_values,
                    )
                )
            for name, value in raw_loss_values.items():
                writer.add_scalar(f"{args.stage}/train/{name}", value, global_iter)

            is_final_iter = epoch == args.epochs - 1 and step == steps_per_epoch
            should_validate = global_iter % args.val_freq == 0 or is_final_iter
            if should_validate and global_iter != last_val_iter:
                psnr, ssim = _validate(args.stage, model, val_loader, device)
                improved = psnr > best_psnr
                if improved:
                    best_psnr, best_ssim = psnr, ssim
                val_logger.info(
                    _format_val_message(
                        args,
                        args.stage,
                        epoch,
                        global_iter,
                        psnr,
                        ssim,
                        best_psnr,
                        best_ssim,
                        improved,
                    )
                )
                writer.add_scalar(f"{args.stage}/val/psnr", psnr, global_iter)
                writer.add_scalar(f"{args.stage}/val/rgb_ssim", ssim, global_iter)
                last_val_iter = global_iter

            periodic = global_iter % args.save_freq == 0 or is_final_iter
            if periodic or should_validate:
                payload = _build_payload(
                    args.stage,
                    model,
                    options,
                    epoch,
                    step,
                    global_iter,
                    best_psnr,
                    best_ssim,
                )
                _save_checkpoint(
                    experiment_dir,
                    args.stage,
                    payload,
                    global_iter,
                    is_best=should_validate and improved,
                    periodic=periodic,
                )
                train_logger.info(
                    f"[{args.experiment}][CHECKPOINT] [stage: {args.stage}] "
                    f"[iter={global_iter:,}] [best_updated={should_validate and improved}]"
                )
        resume_step = 0

    writer.close()


def build_parser():
    """创建单阶段训练命令行参数。"""
    parser = argparse.ArgumentParser(description="URetinex-Net++ 三数据集单阶段训练")
    parser.add_argument("--stage", choices=sorted(STAGE_DEFAULTS), required=True)
    parser.add_argument("--dataset", choices=sorted(LOL_LAYOUTS), required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--print-freq", type=int, default=20)
    parser.add_argument("--val-freq", type=int, default=1000)
    parser.add_argument("--save-freq", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260909)
    return parser
def main():
    """解析参数并补齐与原论文一致的阶段默认超参数。"""
    args = build_parser().parse_args()
    defaults = STAGE_DEFAULTS[args.stage]
    args.epochs = args.epochs or defaults["epochs"]
    args.batch_size = args.batch_size or defaults["batch_size"]
    args.patch_size = args.patch_size or defaults["patch_size"]
    train_stage(args)


if __name__ == "__main__":
    main()
