"""URetinex-Net++ 三数据集复现流程的公共运行时工具。"""

import json
import logging
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
BEIJING_TZ = timezone(timedelta(hours=8))


LOL_LAYOUTS = {
    "lol-v1": {
        "name": "LOL-v1",
        "train_low": "LOL-v1/our485/low",
        "train_high": "LOL-v1/our485/high",
        "test_low": "LOL-v1/eval15/low",
        "test_high": "LOL-v1/eval15/high",
        "train_split": "our485",
        "test_split": "eval15",
    },
    "lol-v2-syn": {
        "name": "LOL-v2-syn",
        "train_low": "LOL-v2/Synthetic/Train/Low",
        "train_high": "LOL-v2/Synthetic/Train/Normal",
        "test_low": "LOL-v2/Synthetic/Test/Low",
        "test_high": "LOL-v2/Synthetic/Test/Normal",
        "train_split": "Synthetic/Train",
        "test_split": "Synthetic/Test",
    },
    "lol-v2-real": {
        "name": "LOL-v2-real",
        "train_low": "LOL-v2/Real_captured/Train/Low",
        "train_high": "LOL-v2/Real_captured/Train/Normal",
        "test_low": "LOL-v2/Real_captured/Test/Low",
        "test_high": "LOL-v2/Real_captured/Test/Normal",
        "train_split": "Real_captured/Train",
        "test_split": "Real_captured/Test",
    },
}


class BeijingFormatter(logging.Formatter):
    """使用北京时间生成训练和验证日志时间戳。"""

    def formatTime(self, record, datefmt=None):  # noqa: N802
        timestamp = datetime.fromtimestamp(record.created, tz=timezone.utc)
        timestamp = timestamp.astimezone(BEIJING_TZ)
        return timestamp.strftime(datefmt or "%Y-%m-%d %H:%M:%S")


def create_logger(name, file_path):
    """创建同时写入终端和指定文件的北京时间 logger。"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    formatter = BeijingFormatter("%(asctime)s %(levelname)s: %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(file_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def dataset_spec(data_root, dataset):
    """返回一个 LOL 子集的训练和测试目录配置。"""
    if dataset not in LOL_LAYOUTS:
        supported = ", ".join(sorted(LOL_LAYOUTS))
        raise ValueError(f"不支持 dataset={dataset!r}，可选值：{supported}。")

    root = Path(data_root).expanduser().resolve()
    spec = dict(LOL_LAYOUTS[dataset])
    for key in ("train_low", "train_high", "test_low", "test_high"):
        spec[key] = root / spec[key]
    return spec


def _relative_key(root, image_path):
    """为成对图像生成规范化且跨平台稳定的相对路径键。"""
    return image_path.relative_to(root).as_posix()


def _index_images(root):
    """递归索引图像，并拒绝重复的规范化相对路径。"""
    if not root.is_dir():
        raise FileNotFoundError(f"图像目录不存在：{root}")

    indexed = {}
    for image_path in sorted(root.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        key = _relative_key(root, image_path)
        if key in indexed:
            raise ValueError(f"目录 {root} 存在重复配对键：{key}")
        indexed[key] = image_path

    if not indexed:
        raise FileNotFoundError(f"目录中未找到支持的图像：{root}")
    return indexed


def build_pairs(low_root, high_root):
    """按规范化相对路径构建并严格校验 LQ/GT 图像对。"""
    low_index = _index_images(Path(low_root))
    high_index = _index_images(Path(high_root))
    low_keys = set(low_index)
    high_keys = set(high_index)
    missing_high = sorted(low_keys - high_keys)
    missing_low = sorted(high_keys - low_keys)
    if missing_high or missing_low:
        messages = []
        if missing_high:
            messages.append(f"缺少 GT {missing_high[:5]}")
        if missing_low:
            messages.append(f"缺少 LQ {missing_low[:5]}")
        raise ValueError("LQ/GT 配对失败：" + "；".join(messages))

    return [(low_index[key], high_index[key], key) for key in sorted(low_keys)]


def _augment_pair(low_image, high_image, mode):
    """以完全相同的几何变换增强一对对齐图像。"""
    if mode == 0:
        return low_image, high_image
    if mode == 1:
        return ImageOps.mirror(low_image), ImageOps.mirror(high_image)
    if mode == 2:
        return ImageOps.flip(low_image), ImageOps.flip(high_image)

    angle = {3: 90, 4: 180, 5: 270, 6: 90, 7: 270}[mode]
    low_image = low_image.rotate(angle, expand=True)
    high_image = high_image.rotate(angle, expand=True)
    if mode in (6, 7):
        low_image = ImageOps.flip(low_image)
        high_image = ImageOps.flip(high_image)
    return low_image, high_image


class PairedImageDataset(Dataset):
    """读取 RGB 成对图像，并在训练时执行对齐裁剪和增强。"""

    def __init__(self, low_root, high_root, patch_size=None, training=False, seed=0):
        self.pairs = build_pairs(low_root, high_root)
        self.patch_size = patch_size
        self.training = training
        self.seed = seed
        self.epoch = 0
        self.to_tensor = ToTensor()

    def set_epoch(self, epoch):
        """设置当前 epoch，使恢复训练后仍得到相同的几何增强。"""
        self.epoch = epoch

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        low_path, high_path, relative_path = self.pairs[index]
        low_image = Image.open(low_path).convert("RGB")
        high_image = Image.open(high_path).convert("RGB")

        if low_image.size != high_image.size:
            raise ValueError(
                f"LQ/GT 尺寸不一致：{low_path} 为 {low_image.size}，"
                f"{high_path} 为 {high_image.size}。"
            )

        if self.training:
            # 按样本和 epoch 固定随机源，避免恢复训练时对齐裁剪发生漂移。
            rng = random.Random(self.seed + self.epoch * 1000003 + index)
            low_image, high_image = _augment_pair(
                low_image, high_image, rng.randint(0, 7)
            )
            if self.patch_size is None:
                raise ValueError("训练数据必须指定 patch_size。")
            width, height = low_image.size
            if width < self.patch_size or height < self.patch_size:
                raise ValueError(
                    f"训练图像 {relative_path} 的尺寸 {height}x{width} 小于 "
                    f"PatchSize={self.patch_size}。"
                )
            left = rng.randint(0, width - self.patch_size)
            top = rng.randint(0, height - self.patch_size)
            crop_box = (left, top, left + self.patch_size, top + self.patch_size)
            low_image = low_image.crop(crop_box)
            high_image = high_image.crop(crop_box)

        return {
            "low_light_img": self.to_tensor(low_image),
            "high_light_img": self.to_tensor(high_image),
            "relative_path": relative_path,
        }


def create_loader(dataset, batch_size, training, num_workers, seed):
    """创建单卡 DataLoader，并固定 sampler 随机源以利于断点复跑。"""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        drop_last=training,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def set_random_seed(seed):
    """设置 Python、NumPy 和 PyTorch 的基础随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # 固定 cuDNN 的卷积选择；不强制 deterministic algorithms，避免原算子直接报错。
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def format_duration(seconds):
    """将秒数转换为可超过 24 小时的 HH:MM:SS。"""
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def tensor_to_rgb255(image_tensor):
    """将单张 RGB `[0,1]` Tensor 转为 HWC `[0,255]` 数组。"""
    image = image_tensor.detach().float().clamp(0, 1).cpu()
    return image.permute(1, 2, 0).numpy() * 255.0


def save_tensor_image(image_tensor, destination):
    """以原始相对路径保存一张裁剪回原尺寸的增强图。"""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image = tensor_to_rgb255(image_tensor)
    Image.fromarray(np.rint(image).clip(0, 255).astype(np.uint8), mode="RGB").save(
        destination
    )


def load_checkpoint(path, map_location="cpu"):
    """兼容 PyTorch 新旧版本读取包含 Namespace 的训练断点。"""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def save_json(payload, destination):
    """保存可阅读的实验配置和运行元数据。"""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, default=str)
