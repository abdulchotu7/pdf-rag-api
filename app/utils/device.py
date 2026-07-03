import torch


def resolve_device(device_setting: str) -> str:
    if device_setting != "auto":
        return device_setting

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
