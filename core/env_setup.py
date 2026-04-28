import os
import logging
import warnings

def initialize_environment():
    """Sets environment variables for ROCm/AMD stability and silences warnings."""
    # Silence Hugging Face
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

    # AMD ROCm 6700 XT Stability Fixes
    os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
    os.environ["MIOPEN_DISABLE_CACHE"] = "1"
    os.environ["PYTORCH_HIP_ALLOC_CONF"] = "garbage_collection_threshold:0.8,max_split_size_mb:512"

    logging.getLogger("transformers").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", message=".*Attempting to use hipBLASLt.*")

def configure_pytorch():
    import torch
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cudnn.enabled = False
    return "cuda" if torch.cuda.is_available() else "cpu"

initialize_environment()