"""Loss functions for SPAD neural reconstruction."""

import torch
import torch.nn.functional as F


def temporal_kl_loss(logits, target_distribution):
    """KL(target || prediction) averaged over batch and pixels."""
    log_probs = F.log_softmax(logits.squeeze(1), dim=1)
    loss_map = F.kl_div(log_probs, target_distribution, reduction="none").sum(dim=1)
    return loss_map.mean()


def depth_tv_loss(depth):
    """Anisotropic total variation on a depth image tensor (B,1,H,W)."""
    if depth.ndim != 4:
        raise ValueError("depth_tv_loss expects shape (B,1,H,W).")
    loss = depth.new_tensor(0.0)
    if depth.shape[-2] > 1:
        loss = loss + torch.mean(torch.abs(depth[..., 1:, :] - depth[..., :-1, :]))
    if depth.shape[-1] > 1:
        loss = loss + torch.mean(torch.abs(depth[..., :, 1:] - depth[..., :, :-1]))
    return loss


def positive_kl_loss(prediction, target, eps=1e-8):
    """KL(target || prediction) for positive photon cubes.

    Both tensors are normalized per sample over all non-batch dimensions.
    """
    pred = torch.clamp(prediction, min=eps)
    tgt = torch.clamp(target, min=eps)
    reduce_dims = tuple(range(1, pred.ndim))
    pred = pred / torch.clamp(pred.sum(dim=reduce_dims, keepdim=True), min=eps)
    tgt = tgt / torch.clamp(tgt.sum(dim=reduce_dims, keepdim=True), min=eps)
    return torch.sum(tgt * (torch.log(tgt) - torch.log(pred)), dim=reduce_dims).mean()


def pukl_loss(measurement, risk_estimate, eps=1e-8):
    """Poisson unbiased KL risk term: -sum y log(risk_estimate)."""
    return -torch.mean(measurement * torch.log(torch.clamp(risk_estimate, min=eps)))
