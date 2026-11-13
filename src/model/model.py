"""Baseline and dual-resolution detectors with the same batch interface."""

import torch
from torch import nn

from .encoders import build_encoder
from .spectral import build_reference


class ClassificationHead(nn.Module):
    def __init__(self, in_features, hidden_features, dropout):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, 2),
        )

    def forward(self, x):
        return self.layers(x)


class BaselineCNN(nn.Module):
    def __init__(self, encoder, hidden_features, dropout):
        super().__init__()
        self.encoder = encoder
        self.head = ClassificationHead(encoder.feature_dim, hidden_features, dropout)

    def forward(self, batch):
        logits = self.head(self.encoder(batch["global"]))
        return {"logits": logits}


class DualResolutionDetector(nn.Module):
    def __init__(
        self,
        global_encoder,
        local_encoder,
        embedding_dim,
        dropout,
        freeze_global=False,
        freeze_local=False,
        reference=None,
    ):
        super().__init__()
        self.global_encoder = global_encoder
        self.local_encoder = local_encoder
        self.reference = reference
        self.freeze_global = freeze_global
        self.freeze_local = freeze_local
        evidence_dim = reference.feature_dim if reference is not None else 0
        self.global_projection = self._projection(
            global_encoder.feature_dim, embedding_dim
        )
        self.local_projection = self._projection(
            local_encoder.feature_dim + evidence_dim, embedding_dim
        )
        self.global_head = ClassificationHead(embedding_dim, embedding_dim, dropout)
        self.local_head = ClassificationHead(embedding_dim, embedding_dim, dropout)
        self.fusion_head = ClassificationHead(2 * embedding_dim, embedding_dim, dropout)
        for module, frozen in (
            (global_encoder, freeze_global),
            (local_encoder, freeze_local),
            (reference, True),
        ):
            if module is not None and frozen:
                module.requires_grad_(False)
        self.train()

    @staticmethod
    def _projection(in_features, out_features):
        return nn.Sequential(
            nn.LayerNorm(in_features), nn.Linear(in_features, out_features), nn.ReLU()
        )

    def train(self, mode=True):
        super().train(mode)
        # Freezing weights alone would still allow BatchNorm buffers to change.
        if self.freeze_global:
            self.global_encoder.eval()
        if self.freeze_local:
            self.local_encoder.eval()
        if self.reference is not None:
            self.reference.eval()
        return self

    def forward(self, batch):
        global_features = self.global_projection(self.global_encoder(batch["global"]))
        mask = batch["local_mask"]
        valid = mask.any(dim=1)
        local_features = global_features.new_zeros(
            (*mask.shape, global_features.shape[1])
        )
        if mask.any():
            patches = batch["local"][mask]
            encoded = self.local_encoder(patches)
            if self.reference is not None:
                with torch.no_grad():
                    evidence = self.reference.evidence(patches)
                encoded = torch.cat([encoded, evidence], dim=1)
            local_features[mask] = self.local_projection(encoded)
        local_features = local_features.sum(dim=1) / mask.sum(
            dim=1, keepdim=True
        ).clamp_min(1)
        global_logits = self.global_head(global_features)
        local_logits = self.local_head(local_features)
        fused_logits = self.fusion_head(
            torch.cat([global_features, local_features], dim=1)
        )
        logits = torch.where(valid[:, None], fused_logits, global_logits)
        return {
            "logits": logits,
            "global_logits": global_logits,
            "local_logits": local_logits,
            "local_valid": valid,
        }


def build_model(config, download_weights=True):
    if config.model.name == "baseline":
        encoder = build_encoder(config.model.encoder, download_weights)
        return BaselineCNN(encoder, config.model.embedding_dim, config.model.dropout)
    if config.model.name == "dual_resolution":
        reference = None
        if config.spectral.objective != "none":
            reference = build_reference(config.spectral)
        return DualResolutionDetector(
            build_encoder(config.model.global_encoder, download_weights),
            build_encoder(config.model.local_encoder, download_weights),
            config.model.embedding_dim,
            config.model.dropout,
            config.model.freeze_global,
            config.model.freeze_local,
            reference,
        )
    raise ValueError(f"Unknown model: {config.model.name}")
