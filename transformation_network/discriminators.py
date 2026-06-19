from typing import Tuple

import flax.linen as nn
import jax.numpy as jnp


class ImageDiscriminator(nn.Module):
    """Image-space discriminator used during AugNet pretraining."""

    widths: Tuple[int, ...] = (16, 32, 64, 128)

    @nn.compact
    def __call__(self, images: jnp.ndarray) -> jnp.ndarray:
        """Return one real/fake logit per input image."""
        x = images
        for i, width in enumerate(self.widths):
            # First layer preserves resolution; later layers downsample.
            stride = 1 if i == 0 else 2
            x = nn.Conv(
                features=width,
                kernel_size=(4, 4),
                strides=(stride, stride),
                padding="SAME",
                name=f"conv_{i}",
            )(x)
            x = nn.leaky_relu(x, negative_slope=0.2)

        x = x.reshape((x.shape[0], -1))
        # Squeeze keeps the discriminator output shape as [batch].
        return nn.Dense(1, name="logit")(x).squeeze(-1)


class FeatureDiscriminator(nn.Module):
    """Feature-space discriminator over frozen classifier features."""

    @nn.compact
    def __call__(self, features: jnp.ndarray) -> jnp.ndarray:
        """Return one real/fake logit per feature vector."""
        return nn.Dense(1, name="logit")(features).squeeze(-1)
