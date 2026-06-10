from __future__ import annotations

import torch
import torch.nn.functional as F


class SymmetricInfoNCELoss(torch.nn.Module):
    def __init__(
        self,
        initial_temperature: float = 0.07,
        max_logit_scale: float = 100.0,
    ) -> None:
        super().__init__()
        self.max_logit_scale = max_logit_scale
        self.logit_scale = torch.nn.Parameter(
            torch.log(torch.tensor(1.0 / initial_temperature))
        )

    def forward(
        self,
        z0: torch.Tensor,
        z1: torch.Tensor,
    ) -> torch.Tensor:
        z0 = F.normalize(z0, dim=-1)
        z1 = F.normalize(z1, dim=-1)
        logit_scale = torch.clamp(
            self.logit_scale.exp(), max=self.max_logit_scale
        )
        logits = logit_scale * (z0 @ z1.T)
        labels = torch.arange(z0.shape[0], device=z0.device)
        loss0 = F.cross_entropy(logits, labels)
        loss1 = F.cross_entropy(logits.T, labels)
        return 0.5 * (loss0 + loss1)

    @property
    def temperature(self) -> float:
        return float(1.0 / self.logit_scale.exp().detach().cpu())


class TemporalContrastiveLoss(torch.nn.Module):
    def __init__(
        self,
        initial_temperature: float = 0.07,
        max_logit_scale: float = 100.0,
    ) -> None:
        super().__init__()
        self.max_logit_scale = max_logit_scale
        self.logit_scale = torch.nn.Parameter(
            torch.log(torch.tensor(1.0 / initial_temperature))
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        positive_mask: torch.Tensor,
    ) -> torch.Tensor:
        if embeddings.ndim != 2:
            raise ValueError("Expected embeddings shape [B, D].")
        if positive_mask.shape != (embeddings.shape[0], embeddings.shape[0]):
            raise ValueError("positive_mask must have shape [B, B].")

        embeddings = F.normalize(embeddings, dim=-1)
        positive_mask = positive_mask.to(device=embeddings.device, dtype=torch.bool)
        eye = torch.eye(embeddings.shape[0], device=embeddings.device, dtype=torch.bool)
        positive_mask = positive_mask & (~eye)

        valid_rows = positive_mask.any(dim=1)
        if not bool(valid_rows.any()):
            return embeddings.sum() * 0.0

        logits = torch.clamp(self.logit_scale.exp(), max=self.max_logit_scale) * (
            embeddings @ embeddings.T
        )
        logits = logits.masked_fill(eye, float("-inf"))
        log_probs = F.log_softmax(logits, dim=1)

        positive_weights = positive_mask.to(dtype=embeddings.dtype)
        positive_weights = positive_weights / positive_weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        selected_log_probs = torch.where(
            positive_mask,
            log_probs,
            torch.zeros_like(log_probs),
        )
        per_row_loss = -(positive_weights * selected_log_probs).sum(dim=1)
        return per_row_loss[valid_rows].mean()

    @property
    def temperature(self) -> float:
        return float(1.0 / self.logit_scale.exp().detach().cpu())


class ModulationRegularizationLoss(torch.nn.Module):
    def forward(
        self,
        gamma: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        gamma_penalty = torch.mean((gamma - 1.0) ** 2)
        beta_penalty = torch.mean(beta**2)
        return gamma_penalty + beta_penalty


class DynamicTotalLoss(torch.nn.Module):
    def __init__(
        self,
        initial_temperature: float = 0.07,
        max_logit_scale: float = 100.0,
        regularization_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.regularization_weight = regularization_weight
        self.temporal = TemporalContrastiveLoss(
            initial_temperature=initial_temperature,
            max_logit_scale=max_logit_scale,
        )
        self.regularization = ModulationRegularizationLoss()

    def forward(
        self,
        projected_embeddings: torch.Tensor,
        positive_mask: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        temporal_loss = self.temporal(projected_embeddings, positive_mask)
        reg_loss = self.regularization(gamma, beta)
        total = temporal_loss + (self.regularization_weight * reg_loss)
        return {
            "loss": total,
            "temporal_loss": temporal_loss,
            "regularization_loss": reg_loss,
        }

    @property
    def temperature(self) -> float:
        return self.temporal.temperature
