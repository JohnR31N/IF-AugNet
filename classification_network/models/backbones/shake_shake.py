from typing import Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp


def _shake_mix(branch_a: jnp.ndarray, branch_b: jnp.ndarray, rng, train: bool) -> jnp.ndarray:
    """Mix two residual branches with Shake-Shake forward/backward noise."""
    if not train:
        # Evaluation uses the deterministic average of both branches.
        return 0.5 * (branch_a + branch_b)

    alpha_rng, beta_rng = jax.random.split(rng)
    shape = (branch_a.shape[0],) + (1,) * (branch_a.ndim - 1)
    # alpha controls the forward mixture; beta controls the backward mixture.
    alpha = jax.random.uniform(alpha_rng, shape, dtype=branch_a.dtype)
    beta = jax.random.uniform(beta_rng, shape, dtype=branch_a.dtype)
    forward = alpha * branch_a + (1.0 - alpha) * branch_b
    backward = beta * branch_a + (1.0 - beta) * branch_b
    # Stop-gradient trick decouples forward and backward mixing coefficients.
    return backward + jax.lax.stop_gradient(forward - backward)


class ShakeBranch(nn.Module):
    """One pre-activation branch inside a Shake-Shake block."""

    features: int
    stride: int = 1

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        """Apply a two-convolution residual branch."""
        x = nn.BatchNorm(use_running_average=not train, name="bn1")(x)
        x = nn.relu(x)
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
        x = nn.Conv(
            features=self.features,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            use_bias=False,
            name="conv2",
        )(x)
        return x


class ShakeShakeBlock(nn.Module):
    """Residual block with two branches mixed by Shake-Shake regularization."""

    features: int
    stride: int = 1

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        """Apply the Shake-Shake block and optional projection shortcut."""
        branch_a = ShakeBranch(self.features, self.stride, name="branch_a")(x, train=train)
        branch_b = ShakeBranch(self.features, self.stride, name="branch_b")(x, train=train)
        shake_rng = self.make_rng("dropout") if train else None
        mixed = _shake_mix(branch_a, branch_b, shake_rng, train=train)

        shortcut = x
        if shortcut.shape[-1] != self.features or self.stride != 1:
            # Projection shortcut matches both channels and spatial resolution.
            shortcut = nn.BatchNorm(use_running_average=not train, name="shortcut_bn")(shortcut)
            shortcut = nn.relu(shortcut)
            shortcut = nn.Conv(
                features=self.features,
                kernel_size=(1, 1),
                strides=(self.stride, self.stride),
                padding="SAME",
                use_bias=False,
                name="shortcut",
            )(shortcut)
        return shortcut + mixed


class ShakeShakeResNet(nn.Module):
    """Shake-Shake ResNet backbone for CIFAR experiments."""

    depth: int = 26
    base_width: int = 32
    num_classes: int = 10

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        train: bool = True,
        return_features: bool = False,
    ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        """Run the Shake-Shake ResNet forward pass."""
        if (self.depth - 2) % 6 != 0:
            raise ValueError("ShakeShakeResNet depth must follow 6n + 2.")

        blocks_per_stage = (self.depth - 2) // 6
        widths = [self.base_width, self.base_width * 2, self.base_width * 4]

        x = nn.Conv(
            features=16,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            use_bias=False,
            name="stem",
        )(x)

        for stage, width in enumerate(widths):
            for block in range(blocks_per_stage):
                # Downsample at the first block of stages after stage 0.
                stride = 2 if stage > 0 and block == 0 else 1
                x = ShakeShakeBlock(
                    features=width,
                    stride=stride,
                    name=f"stage{stage}_block{block}",
                )(x, train=train)

        x = nn.BatchNorm(use_running_average=not train, name="final_bn")(x)
        x = nn.relu(x)
        # Pooled features are consumed by the final named classifier.
        features = jnp.mean(x, axis=(1, 2))
        logits = nn.Dense(features=self.num_classes, name="classifier")(features)

        if return_features:
            return features, logits
        return logits


class ShakeShake26x2x32d(ShakeShakeResNet):
    """Convenience preset for Shake-Shake (26 2x32d)."""

    depth: int = 26
    base_width: int = 32
