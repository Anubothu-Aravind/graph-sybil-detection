"""Hardware detection, environment configuration, and utility helpers for GraphShield."""

import os
import platform
import random
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("graphshield")


def get_system_architecture() -> str:
    """Return system machine architecture (e.g. arm64 or x86_64)."""
    return platform.machine().lower()


def is_apple_silicon() -> bool:
    """Check if host machine is Apple Silicon (M1/M2/M3/M4/etc. on arm64)."""
    return platform.system() == "Darwin" and ("arm" in platform.machine().lower() or "aarch64" in platform.machine().lower())


def get_device_info() -> Dict[str, Any]:
    """Inspect macOS hardware and PyTorch backend support (MPS vs CPU).

    Returns a structured dictionary with device characteristics and friendly labels.
    """
    sys_name = platform.system()
    machine = platform.machine()
    is_mac = sys_name == "Darwin"
    apple_silicon = is_apple_silicon()

    info: Dict[str, Any] = {
        "os": sys_name,
        "os_version": platform.mac_ver()[0] if is_mac else platform.version(),
        "architecture": machine,
        "is_apple_silicon": apple_silicon,
        "is_macos": is_mac,
        "torch_available": False,
        "torch_version": None,
        "mps_available": False,
        "mps_built": False,
        "cuda_available": False,
        "recommended_device": "cpu",
        "device_display_name": "CPU (Standard)",
        "details": "",
    }

    try:
        import torch
        info["torch_available"] = True
        info["torch_version"] = torch.__version__

        mps_built = hasattr(torch.backends, "mps") and torch.backends.mps.is_built()
        mps_avail = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        cuda_avail = torch.cuda.is_available()

        info["mps_built"] = mps_built
        info["mps_available"] = mps_avail
        info["cuda_available"] = cuda_avail

        if mps_avail:
            info["recommended_device"] = "mps"
            info["device_display_name"] = "Apple Silicon GPU (Metal Performance Shaders - MPS)"
            info["details"] = "Accelerated matrix operations using Apple Silicon unified memory GPU."
        elif cuda_avail:
            info["recommended_device"] = "cuda"
            info["device_display_name"] = f"NVIDIA GPU ({torch.cuda.get_device_name(0)})"
            info["details"] = "Accelerated matrix operations using CUDA."
        else:
            info["recommended_device"] = "cpu"
            if apple_silicon:
                info["device_display_name"] = "Apple Silicon CPU (Multithreaded Fallback)"
                info["details"] = "MPS acceleration is not active; running efficiently on ARM64 CPU cores."
            elif is_mac:
                info["device_display_name"] = "Intel Mac CPU"
                info["details"] = "Running on x86_64 Intel CPU using optimized NumPy BLAS."
            else:
                info["device_display_name"] = "Generic CPU"
                info["details"] = "Running on system CPU."
    except ImportError:
        info["details"] = "PyTorch is not installed; running via pure NumPy/SciPy algorithms."
        if apple_silicon:
            info["device_display_name"] = "Apple Silicon (NumPy / SciPy CPU)"
        elif is_mac:
            info["device_display_name"] = "Intel Mac (NumPy / SciPy CPU)"
        else:
            info["device_display_name"] = "CPU (NumPy / SciPy)"

    return info


def get_torch_device(prefer_mps: bool = True):
    """Return torch.device object preferring MPS on Apple Silicon, fallback to CPU.

    Returns None if PyTorch is not available.
    """
    try:
        import torch
        if prefer_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    except ImportError:
        return None


def set_seed(seed: int = 42) -> None:
    """Set random seed across Python random, NumPy, and PyTorch for deterministic results."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # torch.mps.manual_seed is available in recent torch versions
            if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
                torch.mps.manual_seed(seed)
    except ImportError:
        pass


def setup_logger(name: str = "graphshield", level: int = logging.INFO) -> logging.Logger:
    """Configure standard application logging."""
    l = logging.getLogger(name)
    if not l.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
        handler.setFormatter(fmt)
        l.addHandler(handler)
    l.setLevel(level)
    return l
