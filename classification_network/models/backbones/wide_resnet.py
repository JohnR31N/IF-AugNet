from typing import Tuple, Union

import flax.linen as nn
import jax.numpy as jnp


class WideResNetBlock(nn.Module):
    """Pre-activation residual block used by WideResNet."""

    features: int
    stride: int = 1
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        """Apply a WideResNet block with optional dropout."""
        residual = x

        # Pre-activation order: BN-ReLU-Conv.
        x = nn.BatchNorm(use_running_average=not train, name="bn1")(x)
        x = nn.relu(x)
        if residual.shape[-1] != self.features or self.stride != 1:
            # Use activated input for the shortcut, matching WRN practice.
            residual = nn.Conv(
                features=self.features,
                kernel_size=(1, 1),
                strides=(self.stride, self.stride),
                padding="SAME",
                use_bias=False,
                name="shortcut",
            )(x)

        x = nn.Conv(
            features=self.features,
            kernel_size=(3, 3),
            strides=(self.stride, self.stride),
            padding="SAME",
            use_bias=False,
            name="conv1",
        )(x)
        x = nn.BatchNorm(use_running_average=not train, name="bn2")(x)
        x = nn.relu(x)
        if self.dropout_rate > 0:
            # Dropout is active only during training.
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)
        x = nn.Conv(
            features=self.features,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            use_bias=False,
            name="conv2",
        )(x)
        return x + residual


class WideResNet(nn.Module):
    """WideResNet classifier for CIFAR Table 2 experiments."""

    depth: int = 28
    width_multiplier: int = 10
    num_classes: int = 10
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        train: bool = True,
        return_features: bool = False,
    ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        """Run WRN forward pass and optionally return pooled features."""
        if (self.depth - 4) % 6 != 0:
            raise ValueError("WideResNet depth must follow 6n + 4.")

        blocks_per_stage = (self.depth - 4) // 6
        widths = [
            16,
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
            name="stem",
        )(x)

        for stage, width in enumerate(widths[1:]):
            for block in range(blocks_per_stage):
                # Stages after the first downsample at their first block.
                stride = 2 if stage > 0 and block == 0 else 1
                x = WideResNetBlock(
                    features=width,
                    stride=stride,
                    dropout_rate=self.dropout_rate,
                    name=f"stage{stage}_block{block}",
                )(x, train=train)

        x = nn.BatchNorm(use_running_average=not train, name="final_bn")(x)
        x = nn.relu(x)
        # Global average pooling exposes features to influence estimation.
        features = jnp.mean(x, axis=(1, 2))
        logits = nn.Dense(features=self.num_classes, name="classifier")(features)

        if return_features:
            return features, logits
        return logits


class WideResNet28x10(WideResNet):
    """Convenience preset for WideResNet-28-10."""

    depth: int = 28
    width_multiplier: int = 10
