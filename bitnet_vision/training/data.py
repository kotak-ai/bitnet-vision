"""CIFAR-10 loading (official binary release, no pickle) and fast tensor augmentation.

The whole dataset fits in memory as uint8 (150 MB), so batches are sliced from tensors and augmented with
vectorized ops. No torchvision dependency.
"""

from __future__ import annotations

import hashlib
import os
import tarfile
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass

import torch
from torch import Tensor

__all__ = [
    "CIFAR10_URL",
    "CIFAR10_MEAN",
    "CIFAR10_STD",
    "TensorDataset",
    "load_cifar10",
    "normalize",
    "augment",
    "iterate_batches",
]

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
CIFAR10_MD5 = "c32a1d4ab5d03f1284b67883e8d87530"
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
_FOLDER = "cifar-10-batches-bin"
_TRAIN_FILES = [f"data_batch_{i}.bin" for i in range(1, 6)]
_TEST_FILES = ["test_batch.bin"]
_RECORD = 1 + 3 * 32 * 32


@dataclass
class TensorDataset:
    images: Tensor  # uint8, (N, 3, H, W)
    labels: Tensor  # int64, (N,)

    def __len__(self) -> int:
        return self.labels.shape[0]

    def subset(self, n: int | None) -> TensorDataset:
        return self if n is None or n >= len(self) else TensorDataset(self.images[:n], self.labels[:n])


def _md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_and_extract(root: str) -> None:
    os.makedirs(root, exist_ok=True)
    archive = os.path.join(root, "cifar-10-binary.tar.gz")
    if not os.path.exists(archive):
        tmp = archive + ".part"
        urllib.request.urlretrieve(CIFAR10_URL, tmp)
        os.replace(tmp, archive)
    if _md5(archive) != CIFAR10_MD5:
        raise RuntimeError(f"{archive} has the wrong checksum; delete it and retry")
    wanted = {f"{_FOLDER}/{f}" for f in _TRAIN_FILES + _TEST_FILES}
    with tarfile.open(archive, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.name in wanted and m.isfile()]
        for m in members:  # extract only the expected regular files (no path traversal)
            src = tar.extractfile(m)
            dst = os.path.join(root, *m.name.split("/"))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst + ".part", "wb") as out:
                out.write(src.read())
            os.replace(dst + ".part", dst)


def _read_batches(folder: str, files: list[str]) -> TensorDataset:
    blobs = []
    for f in files:
        with open(os.path.join(folder, f), "rb") as fh:
            blob = fh.read()
        if len(blob) == 0 or len(blob) % _RECORD:
            raise ValueError(f"{f} has {len(blob)} bytes, not a whole number of {_RECORD}-byte records; "
                             "delete the cifar-10-batches-bin folder and load again")
        blobs.append(blob)
    raw = torch.frombuffer(bytearray(b"".join(blobs)), dtype=torch.uint8).view(-1, _RECORD)
    labels = raw[:, 0].long()
    images = raw[:, 1:].reshape(-1, 3, 32, 32).clone()
    return TensorDataset(images, labels)


def load_cifar10(root: str, download: bool = True) -> tuple[TensorDataset, TensorDataset]:
    """Return ``(train, test)``: 50,000 and 10,000 images as uint8 tensors.

    ``root`` may contain the ``cifar-10-batches-bin`` folder or the official
    ``cifar-10-binary.tar.gz`` archive; otherwise the archive is downloaded (checksum verified).
    """
    folder = os.path.join(root, _FOLDER)
    if not all(os.path.exists(os.path.join(folder, f)) for f in _TRAIN_FILES + _TEST_FILES):
        if not download and not os.path.exists(os.path.join(root, "cifar-10-binary.tar.gz")):
            raise FileNotFoundError(f"CIFAR-10 binary files not found under {root}")
        _download_and_extract(root)
    return _read_batches(folder, _TRAIN_FILES), _read_batches(folder, _TEST_FILES)


def normalize(images: Tensor, mean=CIFAR10_MEAN, std=CIFAR10_STD) -> Tensor:
    """uint8 (B, C, H, W) -> float32, scaled to [0, 1] then standardized per channel."""
    m = torch.tensor(mean, device=images.device).view(1, -1, 1, 1)
    s = torch.tensor(std, device=images.device).view(1, -1, 1, 1)
    return (images.float() / 255.0 - m) / s


def augment(images: Tensor, pad: int = 4, flip: bool = True, generator: torch.Generator | None = None) -> Tensor:
    """Per-sample random crop (reflect padding) and horizontal flip on a uint8 or float batch."""
    b, _, h, w = images.shape
    if pad > 0:
        padded = torch.nn.functional.pad(images.float(), (pad, pad, pad, pad), mode="reflect")
        dy = torch.randint(0, 2 * pad + 1, (b,), generator=generator).to(images.device)
        dx = torch.randint(0, 2 * pad + 1, (b,), generator=generator).to(images.device)
        rows = (dy[:, None] + torch.arange(h, device=images.device))[:, :, None]  # (B, H, 1)
        cols = (dx[:, None] + torch.arange(w, device=images.device))[:, None, :]  # (B, 1, W)
        idx = torch.arange(b, device=images.device)[:, None, None]
        images = padded.permute(0, 2, 3, 1)[idx, rows, cols].permute(0, 3, 1, 2).to(images.dtype)
    if flip:
        mask = (torch.rand(b, generator=generator) < 0.5).to(images.device)
        images = torch.where(mask[:, None, None, None], images.flip(-1), images)
    return images


def iterate_batches(
    data: TensorDataset, batch_size: int, shuffle: bool, generator: torch.Generator | None = None,
    drop_last: bool = False,
) -> Iterator[tuple[Tensor, Tensor]]:
    n = len(data)
    order = torch.randperm(n, generator=generator) if shuffle else torch.arange(n)
    stop = n - n % batch_size if drop_last else n
    for i in range(0, stop, batch_size):
        idx = order[i : i + batch_size]
        yield data.images[idx], data.labels[idx]
