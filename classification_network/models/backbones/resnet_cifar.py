from typing import Tuple, Union

import flax.linen as nn
import jax.numpy as jnp


class CifarBasicBlock(nn.Module):
    """Basic residual block for CIFAR-sized ResNets."""

    features: int
    stride: int = 1

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        """Apply a two-convolution residual block."""
        residual = x

        # First conv may downsample at stage boundaries.
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
            # Projection shortcut aligns residual shape with the main branch.
            residual = nn.Conv(
                features=self.features,
                kernel_size=(1, 1),
                strides=(self.stride, self.stride),
                padding="SAME",
                use_bias=False,
            )(residual)
            residual = nn.BatchNorm(use_running_average=not train)(residual)

        return nn.relu(x + residual)


class CifarResNet(nn.Module):
    """CIFAR ResNet family with depth 6n+2 and a named classifier head."""

    depth: int = 56
    num_classes: int = 10
    width_multiplier: int = 1

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        train: bool = True,
        return_features: bool = False,
    ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        """Run the CIFAR ResNet forward pass."""
        if (self.depth - 2) % 6 != 0:
            raise ValueError("CIFAR ResNet depth must follow 6n + 2.")

        blocks_per_stage = (self.depth - 2) // 6
        widths = [
            16 * self.width_multiplier,
            32 * self.width_multiplier,
            64 * self.width_multiplier,
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

        for stage, width in enumerate(widths):
            for block in range(blocks_per_stage):
                # Downsample only at the first block of non-initial stages.
                stride = 2 if stage > 0 and block == 0 else 1
                x = CifarBasicBlock(features=width, stride=stride)(x, train=train)

        # Global average pooling creates the feature vector for influence.
        features = jnp.mean(x, axis=(1, 2))
        logits = nn.Dense(features=self.num_classes, name="classifier")(features)

        if return_features:
            return features, logits
        return logits


class ResNet56(CifarResNet):
    """Paper-style CIFAR ResNet-56 preset."""

    depth: int = 56
