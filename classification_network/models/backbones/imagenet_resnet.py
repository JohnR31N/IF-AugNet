from typing import Tuple, Union

import flax.linen as nn
import jax.numpy as jnp


class ImageNetBottleneckBlock(nn.Module):
    """Bottleneck residual block for ImageNet-scale ResNets."""

    bottleneck_width: int
    stride: int = 1

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        """Apply 1x1-3x3-1x1 bottleneck with projection when needed."""
        residual = x
        output_width = self.bottleneck_width * 4

        # Reduce channel dimension before the expensive spatial convolution.
        x = nn.Conv(
            features=self.bottleneck_width,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="SAME",
            use_bias=False,
            name="conv1",
        )(x)
        x = nn.BatchNorm(use_running_average=not train, name="bn1")(x)
        x = nn.relu(x)

        x = nn.Conv(
            features=self.bottleneck_width,
            kernel_size=(3, 3),
            strides=(self.stride, self.stride),
            padding="SAME",
            use_bias=False,
            name="conv2",
        )(x)
        x = nn.BatchNorm(use_running_average=not train, name="bn2")(x)
        x = nn.relu(x)

        # Expand back to the residual output width.
        x = nn.Conv(
            features=output_width,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="SAME",
            use_bias=False,
            name="conv3",
        )(x)
        x = nn.BatchNorm(use_running_average=not train, name="bn3")(x)

        if residual.shape != x.shape:
            # Projection shortcut aligns channels and spatial resolution.
            residual = nn.Conv(
                features=output_width,
                kernel_size=(1, 1),
                strides=(self.stride, self.stride),
                padding="SAME",
                use_bias=False,
                name="shortcut_conv",
            )(residual)
            residual = nn.BatchNorm(use_running_average=not train, name="shortcut_bn")(residual)

        return nn.relu(x + residual)


class ImageNetResNet(nn.Module):
    """Configurable ImageNet ResNet backbone with a named classifier layer."""

    stage_sizes: Tuple[int, int, int, int]
    num_classes: int = 1000
    widths: Tuple[int, int, int, int] = (64, 128, 256, 512)
    stem_width: int = 64

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        train: bool = True,
        return_features: bool = False,
    ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        """Run the ImageNet ResNet forward pass."""
        if len(self.stage_sizes) != 4 or len(self.widths) != 4:
            raise ValueError("ImageNetResNet expects four stage sizes and four widths.")

        # Standard ImageNet stem: 7x7 stride-2 conv followed by max pool.
        x = nn.Conv(
            features=self.stem_width,
            kernel_size=(7, 7),
            strides=(2, 2),
            padding="SAME",
            use_bias=False,
            name="stem_conv",
        )(x)
        x = nn.BatchNorm(use_running_average=not train, name="stem_bn")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")

        for stage, (blocks, width) in enumerate(zip(self.stage_sizes, self.widths)):
            for block in range(blocks):
                # Downsample at the first block of stages after the first.
                stride = 2 if stage > 0 and block == 0 else 1
                x = ImageNetBottleneckBlock(
                    bottleneck_width=width,
                    stride=stride,
                    name=f"stage{stage + 1}_block{block + 1}",
                )(x, train=train)

        # Global average pooling produces features for last-layer influence.
        features = jnp.mean(x, axis=(1, 2))
        logits = nn.Dense(features=self.num_classes, name="classifier")(features)

        if return_features:
            return features, logits
        return logits


class ResNet50(ImageNetResNet):
    """ImageNet ResNet-50 stage-size preset."""

    stage_sizes: Tuple[int, int, int, int] = (3, 4, 6, 3)


class ResNet200(ImageNetResNet):
    """ImageNet ResNet-200 stage-size preset."""

    stage_sizes: Tuple[int, int, int, int] = (3, 24, 36, 3)
