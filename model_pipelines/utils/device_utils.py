"""
Device selection utilities supporting DirectML, CUDA, and CPU.

Single source of truth for picking a torch device. Preference order:
    1. DirectML (torch-directml, AMD/Intel/NVIDIA GPUs on Windows + WSL2)
    2. CUDA (NVIDIA only)
    3. CPU

Usage:
    from model_pipelines.utils.device_utils import get_device, should_pin_memory

    device = get_device()
    model = model.to(device)

    DataLoader(dataset, ..., pin_memory=should_pin_memory())
"""

import os
import torch

# Cache the DirectML probe so we don't re-import / re-query every call
_DML_DEVICE = None
_DML_CHECKED = False


def _try_directml():
    """Return a DirectML torch.device if available, else None.

    DirectML appears to PyTorch as the 'privateuseone' backend.
    Different torch-directml versions expose availability differently,
    so we fall back through a few options.
    """
    global _DML_DEVICE, _DML_CHECKED
    if _DML_CHECKED:
        return _DML_DEVICE
    _DML_CHECKED = True

    try:
        import torch_directml  # type: ignore
    except ImportError:
        return None

    try:
        available = torch_directml.is_available()
    except AttributeError:
        # Older torch-directml versions don't have is_available()
        try:
            available = torch_directml.device_count() > 0
        except Exception:
            available = False
    except Exception:
        available = False

    if available:
        try:
            _DML_DEVICE = torch_directml.device()
        except Exception:
            _DML_DEVICE = None

    return _DML_DEVICE


def get_device() -> torch.device:
    """Return the best available device: DirectML > CUDA > CPU."""
    forced = os.environ.get("MMEB_DEVICE", "").strip().lower()
    if forced:
        if forced == "cpu":
            return torch.device("cpu")
        if forced == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("MMEB_DEVICE=cuda was requested, but CUDA is not available.")
            return torch.device("cuda")
        if forced in {"directml", "dml"}:
            dml = _try_directml()
            if dml is None:
                raise RuntimeError("MMEB_DEVICE=directml was requested, but DirectML is not available.")
            return dml
        raise ValueError("MMEB_DEVICE must be one of: cpu, cuda, directml.")

    dml = _try_directml()
    if dml is not None:
        return dml
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device('cpu')


def is_directml(device) -> bool:
    """True if `device` is the DirectML PrivateUse1 device."""
    if device is None:
        return False
    try:
        return device.type == "privateuseone"
    except AttributeError:
        return False


def should_pin_memory() -> bool:
    """pin_memory only meaningfully accelerates CUDA host->device copies.

    On DirectML / CPU it does nothing useful and can produce warnings,
    so only enable it when CUDA is actually being used.
    """
    return torch.cuda.is_available()


def device_summary() -> str:
    """One-liner describing what we picked, for logging at startup."""
    dev = get_device()
    if is_directml(dev):
        return f"DirectML ({dev})"
    if dev.type == "cuda":
        try:
            name = torch.cuda.get_device_name(dev)
            return f"CUDA: {name} ({dev})"
        except Exception:
            return f"CUDA ({dev})"
    return "CPU"
