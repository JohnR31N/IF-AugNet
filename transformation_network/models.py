from typing import Any, Dict, Tuple

import flax.linen as nn
import jax.numpy as jnp

from paramyield_network.models import ParameterYieldNetwork
from transformation_network.transforms import apply_appearance_transform, apply_spatial_transform


class TransformationDecoder(nn.Module):
    """Decoder G that expands tau into dense spatial and appearance fields."""

    image_size: int = 32
    channels: int = 3
    base_width: int = 128
    widths: Tuple[int, ...] = (64, 32, 16)
    spatial_channels: int = 6
    use_appearance: bool = True

    @nn.compact
    def __call__(self, tau: jnp.ndarray) -> jnp.ndarray:
        """Decode one tau vector per image into per-pixel transform params."""
        output_channels = self.spatial_channels
        if self.use_appearance:
            # Appearance uses a CxC color matrix plus C bias values per pixel.
            output_channels += self.channels * self.channels + self.channels

        # Pick a low-resolution seed map that reaches image_size after upsamples.
        upsample_factor = 2 ** len(self.widths)
        start_size = max(1, (self.image_size + upsample_factor - 1) // upsample_factor)
        x = nn.Dense(start_size * start_size * self.base_width, name="proj")(tau)
        x = nn.relu(x)
        x = x.reshape((tau.shape[0], start_size, start_size, self.base_width))

        for i, width in enumerate(self.widths):
            # Transposed convolutions progressively recover image resolution.
            x = nn.ConvTranspose(
                features=width,
                kernel_size=(4, 4),
                strides=(2, 2),
                padding="SAME",
                name=f"deconv_{i}",
            )(x)
            x = nn.relu(x)

        # Small initialization starts G near the identity transformation.
        x = nn.ConvTranspose(
            features=output_channels,
            kernel_size=(4, 4),
            strides=(1, 1),
            padding="SAME",
            name="out",
            kernel_init=nn.initializers.normal(stddev=1e-3),
            bias_init=nn.initializers.zeros,
        )(x)
        return x[:, : self.image_size, : self.image_size, :]


class AugmentationNetwork(nn.Module):
    """Full learnable augmentation model that combines E and G."""

    image_size: int = 32
    channels: int = 3
    tau_dim: int = 128
    tau_dropout: float = 0.5
    spatial_scale: float = 0.20
    appearance_scale: float = 0.25
    smoothing_kernel: int = 4
    use_appearance: bool = True
    encoder_widths: Tuple[int, ...] = (16, 32, 64, 128)
    decoder_widths: Tuple[int, ...] = (64, 32, 16)
    decoder_base_width: int = 128

    @nn.compact
    def __call__(
        self,
        images: jnp.ndarray,
        train: bool = True,
        return_aux: bool = False,
        tau_override: Any = None,
    ) -> Any:
        """Generate an augmented image and optional transform diagnostics."""
        if tau_override is None:
            # E predicts tau from the current image.
            tau = ParameterYieldNetwork(
                tau_dim=self.tau_dim,
                widths=self.encoder_widths,
                name="encoder",
            )(images)
            # Dropout over tau produces diverse transforms for the same image.
            tau = nn.Dropout(rate=self.tau_dropout, deterministic=not train, name="tau_dropout")(tau)
        else:
            # Visualizers can pass explicit tau values for interpolation studies.
            tau = tau_override
        tau = jnp.nan_to_num(tau, nan=0.0, posinf=1.0, neginf=-1.0)
        # G decodes tau into spatial fields and optional appearance fields.
        fields = TransformationDecoder(
            image_size=self.image_size,
            channels=self.channels,
            base_width=self.decoder_base_width,
            widths=self.decoder_widths,
            use_appearance=self.use_appearance and self.channels > 1,
            name="decoder",
        )(tau)
        fields = jnp.nan_to_num(fields, nan=0.0, posinf=1.0, neginf=-1.0)

        # The first six channels are the 2x2 local affine matrix and 2D bias.
        spatial_params = fields[..., :6]
        spatial_images, sample_grid = apply_spatial_transform(
            images,
            spatial_params,
            spatial_scale=self.spatial_scale,
            smoothing_kernel=self.smoothing_kernel,
        )

        aux: Dict[str, jnp.ndarray] = {
            "tau": tau,
            "fields": fields,
            "sample_grid": sample_grid,
            "spatial_images": spatial_images,
        }

        if self.use_appearance and self.channels > 1:
            # RGB datasets may also learn local color mixing and bias.
            appearance_params = fields[..., 6:]
            augmented, appearance_delta = apply_appearance_transform(
                spatial_images,
                appearance_params,
                appearance_scale=self.appearance_scale,
                smoothing_kernel=self.smoothing_kernel,
            )
            aux["appearance_delta"] = appearance_delta
        else:
            augmented = spatial_images

        if return_aux:
            return augmented, aux
        return augmented


class CIFARAugmentationNetwork(AugmentationNetwork):
    """Convenience AugNet preset for 32x32 RGB images."""

    image_size: int = 32
    channels: int = 3
