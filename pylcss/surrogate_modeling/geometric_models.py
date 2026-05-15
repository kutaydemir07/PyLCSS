# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
Geometry-aware neural surrogate backbones.

Two architectures are provided, both deliberately small and self-contained so
they run on CPU for a one-engineer workstation:

- :class:`GeomDeepONet` -- branch/trunk operator network with a SIREN trunk
  that consumes (xyz, sdf) per node.  Adapted from
  He, Koric, Abueidda, Najafi, Jasiuk (CMAME 2024, arXiv 2403.14788).
- :class:`GINONet` -- "GINO-lite": rasterise an SDF onto a fixed background
  grid, run a small Fourier neural operator on the grid, interpolate the
  predicted field back to per-design query points.  Adapted from Li et al.,
  *Geometry-Informed Neural Operator* (NVIDIA / Caltech, 2023).

Both models share two architectural ideas:

1. **Geometry-in, field-out.** Geometry is fed as SDF (signed distance) at
   query coordinates, not as a fixed mesh -- so changing design parameters
   that change the mesh is fine, as long as the CAD pipeline can produce
   the new mesh and we can SDF-evaluate.
2. **Per-node prediction.** Both architectures output a value at every
   query point passed in, so a single trained model serves any per-design
   mesh resolution.

These modules don't talk to the CAD pipeline themselves; that wiring is in
:mod:`geometric_wrapper`.  Here we only define the PyTorch math.
"""

from __future__ import annotations

import math
from typing import List, Optional

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except (ImportError, OSError):
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    # ------------------------------------------------------------------
    # SIREN -- sinusoidal-activated MLP. Key trick for Geom-DeepONet's trunk:
    # high-frequency periodic activations let the network represent sharp
    # field gradients near geometry boundaries without resorting to a much
    # deeper ReLU stack.
    # ------------------------------------------------------------------
    class SIRENLayer(nn.Module):
        """One sin-activated linear layer with the SIREN initialization."""

        def __init__(self, in_features: int, out_features: int,
                     w0: float = 30.0, is_first: bool = False) -> None:
            super().__init__()
            self.in_features = in_features
            self.w0 = w0
            self.is_first = is_first
            self.linear = nn.Linear(in_features, out_features)
            self._init_weights()

        def _init_weights(self) -> None:
            with torch.no_grad():
                if self.is_first:
                    bound = 1.0 / self.in_features
                else:
                    bound = math.sqrt(6.0 / self.in_features) / self.w0
                self.linear.weight.uniform_(-bound, bound)
                self.linear.bias.zero_()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.sin(self.w0 * self.linear(x))


    class SIRENNet(nn.Module):
        """Stack of SIRENLayers followed by a plain linear head."""

        def __init__(
            self, in_dim: int, hidden_dim: int, out_dim: int,
            n_hidden: int = 4, w0_first: float = 30.0, w0_hidden: float = 1.0,
        ) -> None:
            super().__init__()
            layers: List[nn.Module] = [SIRENLayer(in_dim, hidden_dim, w0=w0_first, is_first=True)]
            for _ in range(n_hidden - 1):
                layers.append(SIRENLayer(hidden_dim, hidden_dim, w0=w0_hidden))
            self.body = nn.Sequential(*layers)
            self.head = nn.Linear(hidden_dim, out_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.head(self.body(x))


    # ------------------------------------------------------------------
    # Geom-DeepONet
    # ------------------------------------------------------------------
    class GeomDeepONet(nn.Module):
        """Branch / trunk operator network with geometry-aware trunk.

        Inputs
        ------
        params : (batch, n_param) -- design parameter vector (broadcast across all query points of a sample)
        query  : (n_query, 3 + 1) -- per-node (xyz, sdf) features

        Output
        ------
        field  : (batch, n_query, out_dim) -- predicted field at every query point

        The branch network turns the parameter vector into a latent vector of
        size ``latent_dim``; the trunk network turns each query point into a
        latent vector of the same size; the inner product of the two gives
        one scalar per query (per output channel).
        """

        def __init__(
            self,
            n_param: int,
            out_dim: int,
            latent_dim: int = 64,
            branch_hidden: int = 128,
            branch_layers: int = 3,
            trunk_hidden: int = 64,
            trunk_layers: int = 4,
        ) -> None:
            super().__init__()
            self.n_param = n_param
            self.out_dim = out_dim
            self.latent_dim = latent_dim

            # Branch: MLP from params -> latent_dim * out_dim. Splitting by
            # output channel makes the inner product per-channel rather than
            # shared, which matters when fields have different magnitudes.
            branch: List[nn.Module] = []
            in_d = max(1, n_param)
            for _ in range(branch_layers - 1):
                branch.append(nn.Linear(in_d, branch_hidden))
                branch.append(nn.Tanh())
                in_d = branch_hidden
            branch.append(nn.Linear(in_d, latent_dim * out_dim))
            self.branch = nn.Sequential(*branch)

            # Trunk: SIREN MLP from (xyz, sdf) -> latent_dim * out_dim.
            self.trunk = SIRENNet(
                in_dim=4, hidden_dim=trunk_hidden, out_dim=latent_dim * out_dim,
                n_hidden=trunk_layers,
            )

            # Output bias per channel.
            self.bias = nn.Parameter(torch.zeros(out_dim))

        def forward(self, params: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
            """Predict field values at every query point.

            ``params`` is ``(batch, n_param)`` and ``query`` is
            ``(n_query, 4)``. We tile the trunk output across the batch and
            do a per-sample inner product. For batch_size=1 (typical at
            inference) this is just a dot product.
            """
            if params.dim() == 1:
                params = params.unsqueeze(0)
            B = params.shape[0]
            Q = query.shape[0]

            b = self.branch(params)  # (B, L * out_dim)
            b = b.view(B, self.latent_dim, self.out_dim)

            t = self.trunk(query)  # (Q, L * out_dim)
            t = t.view(Q, self.latent_dim, self.out_dim)

            # Inner product over latent dim:
            # field[b, q, c] = sum_l b[b, l, c] * t[q, l, c] + bias[c]
            # einsum is cleaner than reshape/matmul here.
            field = torch.einsum("blc,qlc->bqc", b, t) + self.bias
            return field


    # ------------------------------------------------------------------
    # GINO -- Fourier neural operator on a background grid
    # ------------------------------------------------------------------
    class _SpectralConv3d(nn.Module):
        """One Fourier-mode mixing layer for FNO. Mixes the lowest k_modes
        Fourier coefficients per axis; everything above is set to zero.

        Keeping this small (modes=4-8) on purpose so the model trains on
        CPU in minutes, not hours.
        """

        def __init__(self, in_channels: int, out_channels: int, modes: int) -> None:
            super().__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.modes = modes
            scale = 1.0 / (in_channels * out_channels)
            # Complex weights stored as real (..., 2) so PyTorch optim works.
            self.weights = nn.Parameter(
                scale * torch.randn(in_channels, out_channels, modes, modes, modes, 2)
            )

        def _complex_mul(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            # x: (B, in_c, modes, modes, modes) complex
            # w stored as real (in_c, out_c, modes, modes, modes, 2)
            w_complex = torch.complex(w[..., 0], w[..., 1])
            return torch.einsum("bixyz,ioxyz->boxyz", x, w_complex)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B, in_c, R, R, R)
            B, _, R, _, _ = x.shape
            x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])  # (B, in_c, R, R, R//2+1)
            out_ft = torch.zeros(
                B, self.out_channels, R, R, R // 2 + 1,
                dtype=torch.cfloat, device=x.device,
            )
            m = min(self.modes, R // 2)
            out_ft[:, :, :m, :m, :m] = self._complex_mul(
                x_ft[:, :, :m, :m, :m], self.weights[:, :, :m, :m, :m, :]
            )
            return torch.fft.irfftn(out_ft, s=(R, R, R), dim=[-3, -2, -1])


    class GINONet(nn.Module):
        """GINO-lite: SDF-on-grid -> small FNO -> interpolate to query points.

        The model accepts an SDF volume + scalar parameter vector and emits a
        per-grid-cell field. At training/predict time we tri-linearly sample
        the predicted field at the design's actual mesh nodes.

        Inputs
        ------
        sdf  : (B, 1, R, R, R) -- signed distance on a background grid
        params : (B, n_param) -- design parameter vector
        query  : (n_query, 3) -- normalized query coords in [0, 1]^3

        Output
        ------
        field : (B, n_query, out_dim)

        The "interpolate at query points" step is a trilinear sample using
        F.grid_sample so the model stays differentiable end-to-end and the
        loss can be computed against per-FEA-node ground truth.
        """

        def __init__(
            self,
            n_param: int,
            out_dim: int,
            grid_size: int = 32,
            modes: int = 4,
            hidden_channels: int = 16,
            n_fno_layers: int = 3,
        ) -> None:
            super().__init__()
            self.n_param = n_param
            self.out_dim = out_dim
            self.grid_size = grid_size
            self.hidden_channels = hidden_channels

            # Lift SDF (1 ch) + broadcasted params (n_param ch) -> hidden_channels.
            self.lift = nn.Conv3d(1 + max(0, n_param), hidden_channels, kernel_size=1)

            # Stack of FNO layers, each: spectral conv + pointwise conv + GELU.
            self.spectral = nn.ModuleList([
                _SpectralConv3d(hidden_channels, hidden_channels, modes)
                for _ in range(n_fno_layers)
            ])
            self.pointwise = nn.ModuleList([
                nn.Conv3d(hidden_channels, hidden_channels, kernel_size=1)
                for _ in range(n_fno_layers)
            ])

            # Project back to out_dim channels.
            self.project = nn.Conv3d(hidden_channels, out_dim, kernel_size=1)

        def forward(
            self,
            sdf: torch.Tensor,        # (B, 1, R, R, R)
            params: torch.Tensor,     # (B, n_param)
            query: torch.Tensor,      # (n_query, 3) -- normalized [-1, 1]
        ) -> torch.Tensor:
            B, _, R, _, _ = sdf.shape
            # Broadcast params as additional input channels.
            if self.n_param > 0:
                p = params.view(B, self.n_param, 1, 1, 1).expand(B, self.n_param, R, R, R)
                x = torch.cat([sdf, p], dim=1)
            else:
                x = sdf
            x = self.lift(x)

            for spec, pw in zip(self.spectral, self.pointwise):
                x = spec(x) + pw(x)
                x = torch.nn.functional.gelu(x)

            x = self.project(x)  # (B, out_dim, R, R, R)

            # Interpolate to query coords.  grid_sample wants
            # grid: (B, D_out, H_out, W_out, 3) where coords are in [-1, 1].
            n_q = query.shape[0]
            # Tile query across batch.
            q = query.view(1, n_q, 1, 1, 3).expand(B, n_q, 1, 1, 3)
            sampled = torch.nn.functional.grid_sample(
                x, q, mode="bilinear", padding_mode="border", align_corners=True,
            )  # (B, out_dim, n_query, 1, 1)
            return sampled.squeeze(-1).squeeze(-1).permute(0, 2, 1).contiguous()  # (B, n_query, out_dim)

else:
    # Torch unavailable: stub classes so imports don't explode.
    class SIRENLayer:
        pass

    class SIRENNet:
        pass

    class GeomDeepONet:
        pass

    class GINONet:
        pass
