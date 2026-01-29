import os
import sys
import atexit


def setup_gpu_safety():
    """
    Sets TI_VISIBLE_DEVICE based on CUDA_VISIBLE_DEVICES.
    Must be called BEFORE importing Taichi or Genesis.
    """
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        gpu_ids = os.environ["CUDA_VISIBLE_DEVICES"]
        primary_gpu = gpu_ids.split(",")[0]  # Use first GPU if multiple specified
        os.environ["TI_VISIBLE_DEVICE"] = primary_gpu
        print(
            f"[GPU Config] CUDA_VISIBLE_DEVICES={gpu_ids}, setting TI_VISIBLE_DEVICE={primary_gpu}"
        )


def _cleanup_torch_conflicts():
    """
    Removes torch.ops and torch.classes from sys.modules to prevent Blender exit crashes.
    """
    for mod_name in ["torch.ops", "torch.classes"]:
        if mod_name in sys.modules:
            try:
                del sys.modules[mod_name]
            except Exception:
                pass


def register_cleanup_handlers():
    """Registers the atexit handler for torch cleanup."""
    atexit.register(_cleanup_torch_conflicts)
