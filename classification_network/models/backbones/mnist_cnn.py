from typing import Tuple, Union

import flax.linen as nn
import jax.numpy as jnp


class MnistConvNet(nn.Module):
    """Small all-convolutional classifier used for MNIST experiments."""

    num_classes: int = 10
    widths: Tuple[int, ...] = (32, 32, 64, 64)
    feature_dim: int = 128

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        train: bool = True,
        return_features: bool = False,
    ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        """Return logits, or final features plus logits for influence code."""
        for i, width in enumerate(self.widths):
            # Each block is Conv-BN-ReLU, with pooling after blocks 1 and 3.
            x = nn.Conv(
                features=width,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="SAME",
                use_bias=False,
                name=f"conv_{i}",
            )(x)
            x = nn.BatchNorm(use_running_average=not train, name=f"bn_{i}")(x)
            x = nn.relu(x)
            if i in (1, 3):
                x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding="SAME")

        # Flatten spatial features before the named final representation.
        x = x.reshape((x.shape[0], -1))
        features = nn.Dense(self.feature_dim, name="features")(x)
        features = nn.relu(features)
        # Keep the classifier layer name stable for last-layer influence.
        logits = nn.Dense(self.num_classes, name="classifier")(features)

        if return_features:
            return features, logits
        return logits
