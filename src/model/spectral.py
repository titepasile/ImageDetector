"""Real-image reference models for reconstruction and conditional flow matching.

The condition contains only low frequencies. The target and noise live in the
complementary high-frequency subspace. Probe errors are features, NOT likelihoods.
"""

import torch
from torch import nn


def split_frequencies(images, cutoff):
    """Split RGB tensors without resizing. Cutoff is in cycles per pixel."""
    height, width = images.shape[-2:]
    fy = torch.fft.fftfreq(height, device=images.device)[:, None]
    fx = torch.fft.rfftfreq(width, device=images.device)[None, :]
    mask = (fx.square() + fy.square()) <= cutoff**2
    spectrum = torch.fft.rfft2(images, norm="ortho")
    low = torch.fft.irfft2(spectrum * mask, s=(height, width), norm="ortho")
    return low, images - low


class SpectralBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(1, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        z = self.conv1(self.relu(self.norm1(x)))
        z = self.conv2(self.relu(self.norm2(z)))
        return x + z


class SpectralReference(nn.Module):
    def __init__(
        self,
        objective,
        channels=32,
        n_blocks=3,
        cutoff=0.125,
        residual_scale=0.1,
        probe_times=(0.25, 0.5, 0.75),
        probe_seed=123,
    ):
        super().__init__()
        if objective not in ("reconstruction", "flow"):
            raise ValueError("Spectral objective must be reconstruction or flow")
        if not 0 < cutoff < 0.5 or residual_scale <= 0:
            raise ValueError("Use 0 < cutoff < 0.5 and a positive residual_scale")
        if not probe_times or any(not 0 < t < 1 for t in probe_times):
            raise ValueError("Probe times must be strictly between zero and one")
        self.objective = objective
        self.cutoff = cutoff
        self.residual_scale = residual_scale
        self.probe_times = tuple(probe_times)
        self.probe_seed = probe_seed
        # Three condition channels, three state channels, one time channel.
        self.init_conv = nn.Conv2d(7, channels, 3, padding=1)
        self.blocks = nn.Sequential(*[SpectralBlock(channels) for _ in range(n_blocks)])
        self.output_conv = nn.Conv2d(channels, 3, 3, padding=1)
        self.feature_dim = 3 if objective == "reconstruction" else 3 * len(probe_times)

    def forward(self, state, time, condition):
        time_image = time[:, None, None, None].expand(-1, 1, *state.shape[-2:])
        z = torch.cat([state, condition, time_image], dim=1)
        prediction = self.output_conv(self.blocks(self.init_conv(z)))
        # Keep the output in the same subspace as the target and noise.
        return split_frequencies(prediction, self.cutoff)[1]

    def training_loss(self, patches):
        condition, residual = split_frequencies(patches, self.cutoff)
        target = residual / self.residual_scale
        if self.objective == "reconstruction":
            time = torch.zeros(len(patches), device=patches.device)
            prediction = self(torch.zeros_like(target), time, condition)
            return (prediction - target).square().mean()
        noise = split_frequencies(torch.randn_like(target), self.cutoff)[1]
        time = torch.rand(len(patches), device=patches.device)
        t = time[:, None, None, None]
        state = (1 - t) * noise + t * target
        velocity = self(state, time, condition)
        return (velocity - (target - noise)).square().mean()

    def evidence(self, patches):
        """Deterministic per-channel discrepancies, independent of batch order.

        Flow probes use fixed noise and times. They do not solve a density ODE
        and must not be interpreted as calibrated anomaly probabilities.
        """
        condition, residual = split_frequencies(patches, self.cutoff)
        target = residual / self.residual_scale
        if self.objective == "reconstruction":
            time = torch.zeros(len(patches), device=patches.device)
            prediction = self(torch.zeros_like(target), time, condition)
            return (prediction - target).square().mean(dim=(-2, -1)).log1p()
        generator = torch.Generator().manual_seed(self.probe_seed)
        noise = torch.randn((1, *target.shape[1:]), generator=generator).to(target)
        noise = split_frequencies(noise, self.cutoff)[1].expand_as(target)
        features = []
        for value in self.probe_times:
            time = target.new_full((len(patches),), value)
            state = (1 - value) * noise + value * target
            error = self(state, time, condition) - (target - noise)
            features.append(error.square().mean(dim=(-2, -1)).log1p())
        return torch.cat(features, dim=1)

    @torch.inference_mode()
    def sample(self, images, steps=20, seed=0):
        """Euler integration for inspecting conditional samples, not detector scoring."""
        if self.objective != "flow" or steps < 1:
            raise ValueError("Sampling requires a flow reference and steps >= 1")
        condition = split_frequencies(images, self.cutoff)[0]
        generator = torch.Generator().manual_seed(seed)
        noise = torch.randn(images.shape, generator=generator).to(images)
        state = split_frequencies(noise, self.cutoff)[1]
        for step in range(steps):
            time = images.new_full((len(images),), step / steps)
            state = state + self(state, time, condition) / steps
        return condition + self.residual_scale * state


def build_reference(config):
    return SpectralReference(
        config.objective,
        config.channels,
        config.blocks,
        config.cutoff,
        config.residual_scale,
        config.probe_times,
        config.probe_seed,
    )
