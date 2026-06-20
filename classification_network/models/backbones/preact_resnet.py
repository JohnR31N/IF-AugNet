from typing import Sequence, Tuple, Union

import flax.linen as nn
import jax.numpy as jnp


class PreActBlock(nn.Module):
    """Pre-activation residual block used by CIFAR PreAct ResNets."""

    features: int
    stride: int = 1

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        """Apply BN/ReLU before each convolution, then add the shortcut."""
        # The first pre-activation tensor is also used by projection shortcuts.
        out = nn.BatchNorm(use_running_average=not train, name="bn1")(x)
        out = nn.relu(out)

        shortcut = x
        if self.stride != 1 or x.shape[-1] != self.features:
            # PreAct ResNet projections consume the activated input.
            shortcut = nn.Conv(
                features=self.features,
                kernel_size=(1, 1),
                strides=(self.stride, self.stride),
                padding="SAME",
                use_bias=False,
                name="shortcut",
            )(out)

        out = nn.Conv(
            features=self.features,
            kernel_size=(3, 3),
            strides=(self.stride, self.stride),
            padding="SAME",
            use_bias=False,
            name="conv1",
        )(out)
        out = nn.BatchNorm(use_running_average=not train, name="bn2")(out)
        out = nn.relu(out)
        out = nn.Conv(
            features=self.features,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            use_bias=False,
            name="conv2",
        )(out)
        return out + shortcut


class PreActResNet(nn.Module):
    """CIFAR-style PreAct ResNet with a named classifier head."""

    stage_sizes: Sequence[int] = (2, 2, 2, 2)
    widths: Sequence[int] = (64, 128, 256, 512)
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
        widths = [int(width * self.width_multiplier) for width in self.widths]

        # CIFAR variants use a 3x3 stem and avoid ImageNet-style max pooling.
        x = nn.Conv(
            features=widths[0],
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            use_bias=False,
            name="stem",
        )(x)

        for stage, (blocks, width) in enumerate(zip(self.stage_sizes, widths)):
            for block in range(blocks):
                # Downsample at the first block of stages 2-4.
                stride = 2 if stage > 0 and block == 0 else 1
                x = PreActBlock(
                    features=width,
                    stride=stride,
                    name=f"stage_{stage}_block_{block}",
                )(x, train=train)

        x = nn.BatchNorm(use_running_average=not train, name="final_bn")(x)
        x = nn.relu(x)
        features = jnp.mean(x, axis=(1, 2))

        # The named classifier params are consumed by last-layer influence.
        logits = nn.Dense(features=self.num_classes, name="classifier")(features)
        if return_features:
            return features, logits
        return logits


class PreActResNet18(PreActResNet):
    """Standard 2-2-2-2 PreAct ResNet-18 preset for CIFAR images."""

    stage_sizes: Sequence[int] = (2, 2, 2, 2)
