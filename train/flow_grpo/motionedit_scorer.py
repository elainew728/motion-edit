# Copyright (c) MotionEdit Team, Tencent AI Seattle (https://motion-edit.github.io/)
import copy
import math
from typing import Any, Dict, Optional
import numpy as np
import torch
import torch.nn.functional as F
import os

current_parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
try:
    from unimatch.unimatch import UniMatch
except Exception as exc:
    UniMatch = None
    _UNIMATCH_IMPORT_ERROR = exc


DEFAULT_OPTICAL_FLOW_CFG = {
    "ckpt_path": f"{current_parent_directory}/scripts/unimatch/pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth",
    "padding_factor": 32,
    "attn_type": "swin",
    "attn_splits_list": [2, 8],
    "corr_radius_list": [-1, 4],
    "prop_radius_list": [-1, 1],
    "num_reg_refine": 6,
    "bidir": False,
    "model_kwargs": {
        "feature_channels": 128,
        "num_scales": 2,
        "upsample_factor": 4,
        "num_head": 1,
        "ffn_dim_expansion": 4,
        "num_transformer_layers": 6,
        "reg_refine": True,
        "task": "flow",
    },
    "eps": 1e-6,
    "q": 0.4,
    "resize_to": None,
}


class MotionEditScorer:
    """Computes motion edit rewards using UniMatch optical flow estimations."""

    def __init__(self, device: torch.device, cfg: Optional[Dict[str, Any]] = None):
        if UniMatch is None:
            raise ImportError(
                "Cannot import UniMatch. Ensure unimatch is installed and on PYTHONPATH."
            ) from _UNIMATCH_IMPORT_ERROR

        self.device = device
        self.cfg = copy.deepcopy(DEFAULT_OPTICAL_FLOW_CFG)
        if cfg:
            self._update_cfg(cfg)

        model_kwargs = dict(self.cfg.get("model_kwargs", {}))
        self.model = UniMatch(**model_kwargs).to(device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

        self._load_checkpoint(self.cfg.get("ckpt_path"))

        self.padding_factor = int(self.cfg.get("padding_factor", 32))
        self.attn_type = self.cfg.get("attn_type", "swin")
        self.attn_splits_list = self.cfg.get("attn_splits_list", [2, 8])
        self.corr_radius_list = self.cfg.get("corr_radius_list", [-1, 4])
        self.prop_radius_list = self.cfg.get("prop_radius_list", [-1, 1])
        self.num_reg_refine = int(self.cfg.get("num_reg_refine", 6))
        self.bidir = bool(self.cfg.get("bidir", False))
        self.resize_to = self.cfg.get("resize_to", None)
        self.eps = float(self.cfg.get("eps", 1e-6))
        self.q = float(self.cfg.get("q", 0.4))

    def _update_cfg(self, cfg: Dict[str, Any]) -> None:
        for key, value in cfg.items():
            if isinstance(value, dict) and key in self.cfg and isinstance(self.cfg[key], dict):
                self.cfg[key].update(value)
            else:
                self.cfg[key] = value

    def _load_checkpoint(self, ckpt_path: Optional[str]) -> None:
        if not ckpt_path:
            return
        state = torch.load(ckpt_path, map_location="cpu")
        state = state["model"] if isinstance(state, dict) and "model" in state else state
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if len(missing) == 0 and len(unexpected) == 0:
            print(f"[flow] Loaded UniMatch weights from {ckpt_path} (clean match).")
        else:
            print(f"[flow][warn] missing keys ({len(missing)}): {missing[:10]} ...")
            print(f"[flow][warn] unexpected keys ({len(unexpected)}): {unexpected[:10]} ...")

    def _maybe_resize(self, tensor: torch.Tensor, size: Optional[Any]) -> torch.Tensor:
        if size is None:
            return tensor
        if isinstance(size, int):
            h, w = tensor.shape[-2:]
            if h >= w:
                new_h, new_w = size, int(round(size * w / h))
            else:
                new_w, new_h = size, int(round(size * h / w))
            return F.interpolate(tensor, size=(new_h, new_w), mode="bilinear", align_corners=False)
        if isinstance(size, (tuple, list)) and len(size) == 2:
            return F.interpolate(tensor, size=tuple(size), mode="bilinear", align_corners=False)
        return tensor

    def _pad_to_factor(self, tensor: torch.Tensor, factor: int) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
        if factor <= 1:
            return tensor, (0, 0, 0, 0)
        _, _, height, width = tensor.shape
        pad_h = (factor - height % factor) % factor
        pad_w = (factor - width % factor) % factor
        padded = F.pad(tensor, (0, pad_w, 0, pad_h))
        return padded, (0, pad_w, 0, pad_h)

    def _unpad(self, tensor: torch.Tensor, pads: tuple[int, int, int, int]) -> torch.Tensor:
        left, right, top, bottom = pads
        if (right + bottom + left + top) == 0:
            return tensor
        return tensor[..., top: tensor.shape[-2] - bottom, left: tensor.shape[-1] - right]

    def _flow(self, src_01: torch.Tensor, tgt_01: torch.Tensor) -> torch.Tensor:
        src = self._maybe_resize(src_01, self.resize_to)
        tgt = self._maybe_resize(tgt_01, self.resize_to)
        src_pad, pads = self._pad_to_factor(src, self.padding_factor)
        tgt_pad, _ = self._pad_to_factor(tgt, self.padding_factor)

        out = self.model(
            src_pad,
            tgt_pad,
            attn_type=self.attn_type,
            attn_splits_list=self.attn_splits_list,
            corr_radius_list=self.corr_radius_list,
            prop_radius_list=self.prop_radius_list,
            num_reg_refine=self.num_reg_refine,
            pred_bidir_flow=self.bidir,
            task="flow",
        )
        flow = out["flow_preds"][-1]
        return self._unpad(flow, pads)

    def _to_chw2(self, flow: torch.Tensor) -> torch.Tensor:
        if flow.ndim != 3:
            raise ValueError(f"Expected 3D flow tensor, got {tuple(flow.shape)}")
        if flow.shape[0] == 2:
            return flow
        if flow.shape[-1] == 2:
            return flow.permute(2, 0, 1).contiguous()
        raise ValueError(f"Flow tensor must have 2 channels, got shape {tuple(flow.shape)}")

    def _direction_misalignment_single(
        self, flow_pred: torch.Tensor, flow_gt: torch.Tensor, eps: float = 1e-6, mthr: float = 0.002
    ) -> torch.Tensor:
        pred = self._to_chw2(flow_pred)
        gt = self._to_chw2(flow_gt)

        if gt.shape[1:] != pred.shape[1:]:
            gt = F.interpolate(gt.unsqueeze(0), size=pred.shape[1:], mode="bilinear", align_corners=False)[0]

        mag_pred = torch.linalg.vector_norm(pred, ord=2, dim=0).clamp_min(eps)
        mag_gt = torch.linalg.vector_norm(gt, ord=2, dim=0)

        u = pred / mag_pred.clamp_min(eps)
        v = gt / mag_gt.clamp_min(eps)

        cos = (u * v).sum(dim=0).clamp(-1.0, 1.0)
        dir_err = (1.0 - cos).clamp(0, 2) * 0.5

        mask = (mag_gt > mthr).float()
        valid = mask.sum().clamp_min(1.0)

        return (dir_err * mask).sum() / valid

    def _flow_diff(self, flow_pred: torch.Tensor, flow_gt: torch.Tensor) -> torch.Tensor:
        height, width, _ = flow_pred.shape
        diag = math.sqrt(height * height + width * width)
        flow_pred_norm = flow_pred / diag
        flow_gt_norm = flow_gt / diag
        diff = flow_pred_norm - flow_gt_norm
        diff = ((diff.abs() + self.eps) ** self.q).mean(dim=(0, 1, 2))
        return diff

    def _pred_to_tensor(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim == 3:
            tensor = image.unsqueeze(0)
        elif image.ndim == 4:
            tensor = image
        else:
            raise ValueError(f"Prediction tensor must be 3D or 4D, got {tuple(image.shape)}")
        tensor = tensor.float().to(self.device)
        if tensor.max() <= 1.5:
            tensor = tensor * 255.0
        return tensor

    def _image_to_numpy(self, image: Any) -> np.ndarray:
        if isinstance(image, torch.Tensor):
            tensor = image.detach().cpu()
            if tensor.ndim == 4 and tensor.shape[0] == 1:
                tensor = tensor[0]
            if tensor.ndim != 3:
                raise ValueError(f"Expected tensor with shape [3,H,W], got {tuple(tensor.shape)}")
            if tensor.shape[0] == 3:
                tensor = tensor.permute(1, 2, 0)
            array = tensor.numpy()
        else:
            array = np.array(image)

        if array.ndim == 2:
            array = np.stack([array, array, array], axis=-1)
        if array.shape[-1] > 3:
            array = array[..., :3]
        array = array.astype(np.float32)
        if array.max() <= 1.5:
            array = array * 255.0
        return array.astype(np.uint8)

    def _ref_to_tensor(self, image: Any) -> torch.Tensor:
        array = self._image_to_numpy(image)
        tensor = torch.from_numpy(array).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
        return tensor

    @torch.no_grad()
    def __call__(self, pred_image: torch.Tensor, ref_image: Any, gt_image: Any) -> float:
        pred_tensor = self._pred_to_tensor(pred_image)
        ref_tensor = self._ref_to_tensor(ref_image)
        gt_tensor = self._ref_to_tensor(gt_image)

        _, _, height, width = pred_tensor.shape
        if ref_tensor.shape[-2:] != (height, width):
            ref_tensor = F.interpolate(ref_tensor, size=(height, width), mode="bilinear", align_corners=False)
        if gt_tensor.shape[-2:] != (height, width):
            gt_tensor = F.interpolate(gt_tensor, size=(height, width), mode="bilinear", align_corners=False)

        flow_pred = self._flow(ref_tensor, pred_tensor)[0].permute(1, 2, 0)
        flow_gt = self._flow(ref_tensor, gt_tensor)[0].permute(1, 2, 0)

        loss_vec = self._flow_diff(flow_pred, flow_gt)

        flow_pred_norm = flow_pred / math.sqrt(height * height + width * width)
        flow_gt_norm = flow_gt / math.sqrt(height * height + width * width)

        dir_loss = self._direction_misalignment_single(flow_pred_norm, flow_gt_norm)

        pred_c2 = self._to_chw2(flow_pred_norm)
        gt_c2 = self._to_chw2(flow_gt_norm)

        mag_gt_map = torch.linalg.vector_norm(gt_c2, ord=2, dim=0)
        mag_pred_map = torch.linalg.vector_norm(pred_c2, ord=2, dim=0)

        mag_gt_mean = mag_gt_map.mean()
        mag_pred_mean = mag_pred_map.mean()
        move_term = torch.clamp(torch.as_tensor(0.01, device=self.device) + 0.5 * mag_gt_mean - mag_pred_mean, min=0.0)

        final_loss = 0.7 * loss_vec + 0.2 * dir_loss + 0.1 * move_term

        loss_vec_best = self._flow_diff(flow_gt, flow_gt)
        max_loss = torch.tensor(1.0, device=self.device, dtype=final_loss.dtype)

        min_a = loss_vec_best.detach()
        max_a = max_loss.detach()
        rng = (max_a - min_a).clamp_min(1e-6)

        norm = ((final_loss - min_a) / rng).clamp(0.0, 1.0)
        inv = 1.0 - norm
        inv = inv.clamp(0.0, 1.0)

        reward_float_0_5 = 5.0 * inv
        final_bins = torch.round(reward_float_0_5).clamp(0, 5)
        final_reward = (final_bins / 5.0).item()

        return float(final_reward)
