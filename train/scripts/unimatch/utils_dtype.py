import torch
import torch.nn.functional as F

def to_like(x, ref):
    # match dtype & device
    return x.to(dtype=ref.dtype, device=ref.device)

def grid_sample_dtype_safe(img, grid, mode='bilinear', padding_mode='zeros', align_corners=True):
    # 1) match dtype/device
    if (grid.dtype != img.dtype) or (grid.device != img.device):
        grid = grid.to(dtype=img.dtype, device=img.device)
    try:
        return F.grid_sample(img, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
    except RuntimeError as e:
        # fallback for builds without bf16 grid_sample
        if img.dtype == torch.bfloat16 and "expected scalar type" in str(e):
            out = F.grid_sample(img.float(), grid.float(), mode=mode, padding_mode=padding_mode, align_corners=align_corners)
            return out.to(torch.bfloat16)
        raise
