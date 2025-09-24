"""
Deterministic data loading & augmentation shared by all experiments.
Covers ImageNet-C, Tiny-ImageNet-C, ImageNet-V2/A, DomainNet and the
ImageNet-VID-C video stream used in Experiment-3.
"""
from __future__ import annotations

import glob
import os
from typing import Tuple

import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


def _transform(size: int = 224):
    return transforms.Compose(
        [
            transforms.Resize(size, interpolation=Image.BICUBIC),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD),
        ]
    )


# --------------------------------------------------------------------------------------
# Static image datasets (Experiments-1/2)
# --------------------------------------------------------------------------------------

def imagenet_c_loader(corruption: str, severity: int, batch_size: int, num_workers: int, *, subset: int | None = None):
    hf_id = "ang9867/ImageNet-C"
    split = f"{corruption}_{severity}"
    ds = load_dataset(hf_id, split=split)
    if subset is not None:
        ds = ds.select(range(subset))
    ds = ds.with_format("torch")
    tfm = _transform(224)

    def _map(batch):
        batch["pixel_values"] = [tfm(img.convert("RGB")) for img in batch["image"]]
        return batch

    ds = ds.map(_map, batched=True)

    def _collate(batch):
        return (
            torch.stack([x["pixel_values"] for x in batch]),
            torch.tensor([x["label"] for x in batch]),
        )

    return (
        DataLoader(ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=False, collate_fn=_collate),
        len(ds),
    )


def tiny_imagenet_c_loader(severity: int, batch_size: int, num_workers: int, *, subset: int | None = None):
    ds = load_dataset("randall-lab/tiny-imagenet-c", split="test", trust_remote_code=True)
    if subset is not None:
        ds = ds.select(range(subset))
    ds = ds.filter(lambda x: x["severity"] == severity)
    ds = ds.with_format("torch")
    tfm = _transform(64)

    def _map(batch):
        batch["pixel_values"] = [tfm(img.convert("RGB")) for img in batch["image"]]
        return batch

    ds = ds.map(_map, batched=True)

    def _collate(batch):
        return (
            torch.stack([x["pixel_values"] for x in batch]),
            torch.tensor([x["label"] for x in batch]),
        )

    return (
        DataLoader(ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=False, collate_fn=_collate),
        len(ds),
    )


def imagenet_v2_loader(batch_size: int, num_workers: int, *, subset: int | None = None):
    ds = load_dataset("djghosh/wds_imagenetv2_test", split="test")
    if subset is not None:
        ds = ds.select(range(subset))
    ds = ds.with_format("torch")
    tfm = _transform(224)
    ds = ds.map(lambda b: {"pixel_values": [tfm(img.convert("RGB")) for img in b["image"]]}, batched=True)

    def _collate(batch):
        return (
            torch.stack([x["pixel_values"] for x in batch]),
            torch.tensor([x["label"] for x in batch]),
        )

    return (
        DataLoader(ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=False, collate_fn=_collate),
        len(ds),
    )


def imagenet_a_loader(batch_size: int, num_workers: int, *, subset: int | None = None):
    ds = load_dataset("barkermrl/imagenet-a", split="test")
    if subset is not None:
        ds = ds.select(range(subset))
    ds = ds.with_format("torch")
    tfm = _transform(224)
    ds = ds.map(lambda b: {"pixel_values": [tfm(img.convert("RGB")) for img in b["image"]]}, batched=True)

    def _collate(batch):
        return (
            torch.stack([x["pixel_values"] for x in batch]),
            torch.tensor([x["label"] for x in batch]),
        )

    return (
        DataLoader(ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=False, collate_fn=_collate),
        len(ds),
    )


def domainnet_loader(domain: str, batch_size: int, num_workers: int, *, subset: int | None = None):
    hf = {
        "clipart": "Bruece/domainnet-126-by-class-clipart",
        "sketch": "Bruece/domainnet-126-by-class-sketch",
    }.get(domain)
    if hf is None:
        raise ValueError(f"Unknown DomainNet domain '{domain}'")

    ds = load_dataset(hf, split="test")
    if subset is not None:
        ds = ds.select(range(subset))
    ds = ds.with_format("torch")
    tfm = _transform(224)
    ds = ds.map(lambda b: {"pixel_values": [tfm(img.convert("RGB")) for img in b["image"]]}, batched=True)

    def _collate(batch):
        return (
            torch.stack([x["pixel_values"] for x in batch]),
            torch.tensor([x["label"] for x in batch]),
        )

    return (
        DataLoader(ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=False, collate_fn=_collate),
        len(ds),
    )

# --------------------------------------------------------------------------------------
# Streaming video iterator (Experiment-3)
# --------------------------------------------------------------------------------------

def imagenet_vid_c_iter(*, batch_size: int = 1, subset_frames: int | None = None):
    """Generator yielding single-frame tensors (N=1) from ImageNet-VID-C."""

    try:
        from decord import VideoReader, cpu
    except ImportError as e:  # soft-fail – explicit message
        raise ImportError("decord is required for video streaming experiments") from e

    root = load_dataset("ang9867/ImageNet-C", split="original").cache_files[0]["filename"]
    vid_files = glob.glob(os.path.join(os.path.dirname(root), "ILSVRC2015_val_0001_*.mp4"))
    tfm = _transform(224)

    cnt = 0
    for vf in vid_files:
        vr = VideoReader(vf, ctx=cpu(0))
        for frame in vr:
            if subset_frames and cnt >= subset_frames:
                return
            img = Image.fromarray(frame.asnumpy())
            yield tfm(img).unsqueeze(0), torch.tensor(-1)  # label unused
            cnt += 1