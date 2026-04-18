"""Central hardware detection — single source of truth for device selection.

Every module imports from here instead of reading DEVICE from config.
Automatically prefers GPU (CUDA) and falls back to CPU.
"""

from __future__ import annotations

import logging
import os
import platform

import torch

logger = logging.getLogger(__name__)

_detected_device: torch.device | None = None
_banner_printed = False


def detect_device() -> torch.device:
    """Detect the best available device (GPU preferred, CPU fallback).

    Returns the same cached device on subsequent calls.
    Prints a hardware summary banner on first invocation.
    """
    global _detected_device, _banner_printed

    if _detected_device is not None:
        return _detected_device

    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram_mb = torch.cuda.get_device_properties(0).total_mem / (1024 ** 2)
        device_info = f"GPU: {gpu_name} ({vram_mb:.0f} MB VRAM)"
    else:
        device = torch.device("cpu")
        device_info = f"CPU: {platform.processor() or platform.machine()}"

    _detected_device = device

    if not _banner_printed:
        _banner_printed = True
        logger.info("=" * 60)
        logger.info("  HARDWARE DETECTION")
        logger.info("  PyTorch version : %s", torch.__version__)
        logger.info("  CUDA available  : %s", torch.cuda.is_available())
        logger.info("  Selected device : %s", device)
        logger.info("  Device info     : %s", device_info)
        logger.info("  OS              : %s", platform.system())
        logger.info("=" * 60)

    return device


def get_optimal_workers() -> int:
    """Return optimal DataLoader num_workers for the current platform.

    Windows: 0 (multiprocessing spawn is slow and error-prone)
    Linux/Mac: min(4, cpu_count)
    """
    if platform.system() == "Windows":
        return 0
    cpu_count = os.cpu_count() or 1
    return min(4, cpu_count)


def get_optimal_batch_size(device: torch.device | None = None) -> int:
    """Return a reasonable default batch size for the detected device.

    GPU: 64 (fits comfortably in most VRAM)
    CPU: 32 (avoids excessive memory pressure)
    """
    if device is None:
        device = detect_device()
    return 64 if device.type == "cuda" else 32


def get_pin_memory(device: torch.device | None = None) -> bool:
    """Return whether to use pin_memory in DataLoader (True for CUDA)."""
    if device is None:
        device = detect_device()
    return device.type == "cuda"
