from typing import Tuple

import flax.linen as nn
import jax.numpy as jnp


class ParameterYieldNetwork(nn.Module):
    """Encoder E that maps an input image to the latent augmentation code tau."""

    tau_dim: int = 128
    widths: Tuple[int, ...] = (16, 32, 64, 128)

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Produce a bounded tau vector for each input image."""
        for i, width in enumerate(self.widths):
            # The first block preserves resolution; later blocks downsample.
            stride = 1 if i == 0 else 2
            x = nn.Conv(
                features=width,
                kernel_size=(4, 4),
                strides=(stride, stride),
                padding="SAME",
                name=f"conv_{i}",
            )(x)
            x = nn.relu(x)

        # Flatten spatial features before predicting the compact tau code.
        x = x.reshape((x.shape[0], -1))
        # tanh keeps tau bounded, matching the paper's latent transform code.
        return nn.tanh(nn.Dense(self.tau_dim, name="tau")(x))


# Backward-friendly alias used in paper terminology and older scripts.
AugmentationEncoder = ParameterYieldNetwork
