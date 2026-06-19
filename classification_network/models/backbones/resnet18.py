from typing import Sequence, Tuple, Union

import flax.linen as nn
import jax.numpy as jnp


class BasicBlock(nn.Module):
    """Two-convolution residual block for the small ResNet-18 backbone."""

    features: int
    stride: int = 1

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        """Apply residual block and projection shortcut when shapes differ."""
        residual = x

        # First conv may downsample spatial resolution through stride.
        x = nn.Conv(
            features=self.features,
            kernel_size=(3, 3),
            strides=(self.stride, self.stride),
            padding="SAME",
            use_bias=False,
        )(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)

        x = nn.Conv(
            features=self.features,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            use_bias=False,
        )(x)
        x = nn.BatchNorm(use_running_average=not train)(x)

        if residual.shape != x.shape:
            # Match shortcut channels/resolution before adding residual.
            residual = nn.Conv(
                features=self.features,
                kernel_size=(1, 1),
                strides=(self.stride, self.stride),
                padding="SAME",
                use_bias=False,
            )(residual)
            residual = nn.BatchNorm(use_running_average=not train)(residual)

        x = x + residual
        x = nn.relu(x)
        return x


class ResNet18(nn.Module):
    """Compact ResNet-18 classifier used by smoke tests and debug runs."""

    num_classes: int = 10
    width_multiplier: int = 1

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        train: bool = True,
        return_features: bool = False,
    ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        """Return logits, or pooled features plus logits for influence code."""
        widths = [
            64 * self.width_multiplier,
            128 * self.width_multiplier,
            256 * self.width_multiplier,
            512 * self.width_multiplier,
        ]

        x = nn.Conv(
            features=widths[0],
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            use_bias=False,
        )(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)

        # Four residual stages, with downsampling at the first block of stages 2-4.
        x = BasicBlock(features=widths[0], stride=1)(x, train=train)
        x = BasicBlock(features=widths[0], stride=1)(x, train=train)

        x = BasicBlock(features=widths[1], stride=2)(x, train=train)
        x = BasicBlock(features=widths[1], stride=1)(x, train=train)

        x = BasicBlock(features=widths[2], stride=2)(x, train=train)
        x = BasicBlock(features=widths[2], stride=1)(x, train=train)

        x = BasicBlock(features=widths[3], stride=2)(x, train=train)
        x = BasicBlock(features=widths[3], stride=1)(x, train=train)

        features = jnp.mean(x, axis=(1, 2))

        # The named classifier params are consumed by last-layer influence.
        logits = nn.Dense(
            features=self.num_classes,
            name="classifier",
        )(features)

        if return_features:
            return features, logits

        return logits
