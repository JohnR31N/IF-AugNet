import flax.linen as nn
import jax.numpy as jnp


class LinearHead(nn.Module):
    """Named final classifier head used by influence estimation."""

    num_classes: int
    use_bias: bool = True

    @nn.compact
    def __call__(self, features: jnp.ndarray) -> jnp.ndarray:
        """Project feature vectors to class logits."""
        # The name "classifier" is used to find the final-layer params.
        return nn.Dense(
            features=self.num_classes,
            use_bias=self.use_bias,
            name="classifier",
        )(features)
