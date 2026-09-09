"""The denoising autoencoder: a CNN or GRU encoder, a transposed-conv decoder.

Window size 1 (the paper's default) reduces the CNN encoder to an MLP and the GRU to a
single nonlinear layer; the conv/GRU machinery stays in place so the ablation over window
sizes in the paper still runs through the same code path.
"""

from __future__ import annotations

import itertools
import math
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn


def _conv1d_out_length(length: int, kernel_size: int, stride: int, padding: int) -> int:
    return (length + 2 * padding - (kernel_size - 1) - 1) // stride + 1


def _convtranspose1d_out_length(length: int, kernel_size: int, stride: int, padding: int, output_padding: int = 0) -> int:
    return (length - 1) * stride - 2 * padding + (kernel_size - 1) + output_padding + 1


def _auto_paddings(kernel_sizes: Iterable[int]) -> list[int]:
    return [(int(k) - 1) // 2 for k in kernel_sizes]


def _decoder_length(base_length: int, kernel_sizes, strides, paddings, output_paddings) -> int:
    length = base_length
    for k, s, p, op in zip(kernel_sizes, strides, paddings, output_paddings):
        length = _convtranspose1d_out_length(length, k, s, p, op)
    return length


def _best_output_paddings(base_length: int, target_length: int, kernel_sizes, strides, paddings) -> tuple[list[int], int]:
    """ConvTranspose1d output length depends on output_padding; search the small space of
    valid combinations (each in [0, stride)) for the one that lands closest to target_length."""
    best, best_len = None, -1
    for combo in itertools.product(*(range(s) for s in strides)):
        out_len = _decoder_length(base_length, kernel_sizes, strides, paddings, combo)
        if out_len >= target_length and (best is None or out_len < best_len):
            best, best_len = list(combo), out_len
        elif best is None and out_len > best_len:
            best, best_len = list(combo), out_len
    return best or [0] * len(strides), best_len


def _decoder_base_length(target_length: int, kernel_sizes, strides, paddings) -> tuple[int, list[int], int]:
    """Grows the decoder's starting sequence length until some output_padding combo can hit target_length."""
    base_length = max(1, math.ceil(target_length / math.prod(strides))) if strides else target_length
    for _ in range(target_length + 1):
        pad, out_len = _best_output_paddings(base_length, target_length, kernel_sizes, strides, paddings)
        if out_len >= target_length:
            return base_length, pad, out_len
        base_length += 1
    return base_length, pad, out_len


class ConvEncoder(nn.Module):
    def __init__(self, *, input_dim: int, channels: list[int], kernel_sizes: list[int], strides: list[int]) -> None:
        super().__init__()
        paddings = _auto_paddings(kernel_sizes)
        blocks = []
        in_ch = input_dim
        for out_ch, k, s, p in zip(channels, kernel_sizes, strides, paddings):
            blocks += [nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=s, padding=p), nn.GELU()]
            in_ch = out_ch
        self.net = nn.Sequential(*blocks)
        self.kernel_sizes, self.strides, self.paddings = kernel_sizes, strides, paddings

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def output_length(self, window_size: int) -> int:
        length = window_size
        for k, s, p in zip(self.kernel_sizes, self.strides, self.paddings):
            length = _conv1d_out_length(length, k, s, p)
        return length


class AutoencoderModel(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        window_size: int,
        latent_dim: int,
        encoder_type: str,
        cnn_cfg: dict | None,
        gru_cfg: dict | None,
        decoder_cfg: dict,
    ) -> None:
        super().__init__()
        self.input_dim, self.window_size, self.latent_dim = input_dim, window_size, latent_dim
        self.encoder_type = encoder_type.strip().lower()
        if self.encoder_type not in ("cnn", "gru"):
            raise ValueError(f"encoder_type must be 'cnn' or 'gru', got {encoder_type!r}")

        kernel_sizes = list(decoder_cfg.get("kernel_sizes", []))
        strides = list(decoder_cfg.get("strides", []))
        dec_hidden = list(decoder_cfg.get("channels", []))
        paddings = _auto_paddings(kernel_sizes)
        base_channels = decoder_cfg.get("base_channels", dec_hidden[0] if dec_hidden else 128)

        if self.encoder_type == "cnn":
            enc_channels = list(cnn_cfg["channels"])
            self.encoder = ConvEncoder(
                input_dim=input_dim, channels=enc_channels, kernel_sizes=list(cnn_cfg["kernel_sizes"]), strides=list(cnn_cfg["strides"])
            )
            self.base_length = self.encoder.output_length(window_size)
            base_channels = enc_channels[-1]
            self.fc_encode = nn.Linear(base_channels * self.base_length, latent_dim)
        else:
            hidden_dim = int(gru_cfg.get("hidden_dim", 64))
            self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=int(gru_cfg.get("num_layers", 1)), batch_first=True)
            self.fc_encode = nn.Linear(hidden_dim, latent_dim)
            self.base_length, out_paddings, _ = _decoder_base_length(window_size, kernel_sizes, strides, paddings)
            self._decoder_output_paddings = out_paddings

        self.base_channels = base_channels
        self.fc_decode = nn.Linear(latent_dim, base_channels * self.base_length)

        if self.encoder_type == "cnn":
            self._decoder_output_paddings, _ = _best_output_paddings(self.base_length, window_size, kernel_sizes, strides, paddings)

        dec_channels = [base_channels] + dec_hidden + [input_dim]
        layers = []
        for i, (k, s, p, op) in enumerate(zip(kernel_sizes, strides, paddings, self._decoder_output_paddings)):
            layers.append(nn.ConvTranspose1d(dec_channels[i], dec_channels[i + 1], kernel_size=k, stride=s, padding=p, output_padding=op))
            if i < len(kernel_sizes) - 1:
                layers.append(nn.GELU())
        self.decoder = nn.Sequential(*layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.encoder_type == "cnn":
            enc = self.encoder(x.transpose(1, 2))
            return self.fc_encode(enc.reshape(enc.size(0), -1))
        _, h_n = self.gru(x)
        return self.fc_encode(h_n[-1])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        dec = self.fc_decode(z).view(z.size(0), self.base_channels, self.base_length)
        recon = self.decoder(dec)
        if recon.size(2) > self.window_size:
            recon = recon[:, :, : self.window_size]
        elif recon.size(2) < self.window_size:
            recon = F.pad(recon, (0, self.window_size - recon.size(2)))
        return recon.transpose(1, 2), z
