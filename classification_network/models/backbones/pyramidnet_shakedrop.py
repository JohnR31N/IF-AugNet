from typing import Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp


def _shake_drop(x: jnp.ndarray, rng, train: bool, keep_prob: float) -> jnp.ndarray:
    """Apply ShakeDrop stochastic residual scaling."""
    if not train:
        # At eval time, use the expected residual magnitude.
        return keep_prob * x

    gate_rng, alpha_rng, beta_rng = jax.random.split(rng, 3)
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    # gate drops or keeps each sample's residual path.
    gate = jax.random.bernoulli(gate_rng, keep_prob, shape).astype(x.dtype)
    alpha = jax.random.uniform(alpha_rng, shape, minval=-1.0, maxval=1.0, dtype=x.dtype)
    beta = jax.random.uniform(beta_rng, shape, minval=0.0, maxval=1.0, dtype=x.dtype)
    forward_coeff = gate + alpha - gate * alpha
    backward_coeff = gate + beta - gate * beta
    forward = forward_coeff * x
    backward = backward_coeff * x
    # Stop-gradient separates stochastic forward and backward coefficients.
    return backward + jax.lax.stop_gradient(forward - backward)


def _match_shortcut(x: jnp.ndarray, out_channels: int, stride: int) -> jnp.ndarray:
    """Downsample and zero-pad shortcuts to match PyramidNet channels."""
    if stride > 1:
        # Average pooling is the shortcut downsampling used by PyramidNet.
        x = nn.avg_pool(
            x,
            window_shape=(2, 2),
            strides=(stride, stride),
            padding="VALID",
        )
    channels = x.shape[-1]
    if channels < out_channels:
        # Pad channels symmetrically instead of learning a projection.
        pad = out_channels - channels
        left = pad // 2
        right = pad - left
        x = jnp.pad(x, ((0, 0), (0, 0), (0, 0), (left, right)))
    return x


class PyramidNetShakeDropBlock(nn.Module):
    """PyramidNet residual block with optional bottleneck and ShakeDrop."""

    out_channels: int
    stride: int = 1
    keep_prob: float = 1.0
    bottleneck: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        """Apply one PyramidNet block and stochastic residual scaling."""
        shortcut = _match_shortcut(x, self.out_channels, self.stride)

        y = nn.BatchNorm(use_running_average=not train, name="bn1")(x)
        y = nn.relu(y)
        if self.bottleneck:
            # Bottleneck block: 1x1 reduce, 3x3 spatial, 1x1 expand.
            inner_channels = max(self.out_channels // 4, 1)
            y = nn.Conv(
                features=inner_channels,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding="SAME",
                use_bias=False,
                name="conv1",
            )(y)
            y = nn.BatchNorm(use_running_average=not train, name="bn2")(y)
            y = nn.relu(y)
            y = nn.Conv(
                features=inner_channels,
                kernel_size=(3, 3),
                strides=(self.stride, self.stride),
                padding="SAME",
                use_bias=False,
                name="conv2",
            )(y)
            y = nn.BatchNorm(use_running_average=not train, name="bn3")(y)
            y = nn.relu(y)
            y = nn.Conv(
                features=self.out_channels,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding="SAME",
                use_bias=False,
                name="conv3",
            )(y)
        else:
            # Basic block variant used for shallower PyramidNet configs.
            y = nn.Conv(
                features=self.out_channels,
                kernel_size=(3, 3),
                strides=(self.stride, self.stride),
                padding="SAME",
                use_bias=False,
                name="conv1",
            )(y)
            y = nn.BatchNorm(use_running_average=not train, name="bn2")(y)
            y = nn.relu(y)
            y = nn.Conv(
                features=self.out_channels,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="SAME",
                use_bias=False,
                name="conv2",
            )(y)

        shake_rng = self.make_rng("dropout") if train else None
        # ShakeDrop applies to the residual branch before adding shortcut.
        y = _shake_drop(y, shake_rng, train=train, keep_prob=self.keep_prob)
        return shortcut + y


class PyramidNetShakeDrop(nn.Module):
    """PyramidNet backbone with linearly increasing channels and ShakeDrop."""

    depth: int = 272
    alpha: int = 200
    num_classes: int = 10
    bottleneck: bool = True
    final_keep_prob: float = 0.5

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        train: bool = True,
        return_features: bool = False,
    ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        """Run PyramidNet and optionally return pooled features."""
        if self.bottleneck:
            if (self.depth - 2) % 9 != 0:
                raise ValueError("Bottleneck PyramidNet depth must follow 9n + 2.")
            blocks_per_stage = (self.depth - 2) // 9
        else:
            if (self.depth - 2) % 6 != 0:
                raise ValueError("Basic PyramidNet depth must follow 6n + 2.")
            blocks_per_stage = (self.depth - 2) // 6

        total_blocks = blocks_per_stage * 3
        # PyramidNet increases channel count gradually across all blocks.
        add_rate = self.alpha / total_blocks
        block_index = 0

        x = nn.Conv(
            features=16,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            use_bias=False,
            name="stem",
        )(x)

        for stage in range(3):
            for block in range(blocks_per_stage):
                block_index += 1
                out_channels = int(round(16 + add_rate * block_index))
                stride = 2 if stage > 0 and block == 0 else 1
                progress = block_index / total_blocks
                # Later blocks use lower keep probability for stronger ShakeDrop.
                keep_prob = 1.0 - progress * (1.0 - self.final_keep_prob)
                x = PyramidNetShakeDropBlock(
                    out_channels=out_channels,
                    stride=stride,
                    keep_prob=keep_prob,
                    bottleneck=self.bottleneck,
                    name=f"stage{stage}_block{block}",
                )(x, train=train)

        x = nn.BatchNorm(use_running_average=not train, name="final_bn")(x)
        x = nn.relu(x)
        # Final pooled features feed the named classifier layer.
        features = jnp.mean(x, axis=(1, 2))
        logits = nn.Dense(features=self.num_classes, name="classifier")(features)

        if return_features:
            return features, logits
        return logits


class PyramidNet272ShakeDrop(PyramidNetShakeDrop):
    """Convenience preset for PyramidNet-272 with ShakeDrop."""

    depth: int = 272
    alpha: int = 200
    bottleneck: bool = True
