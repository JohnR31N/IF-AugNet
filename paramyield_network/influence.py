from typing import Any, Callable, Dict

import jax
import jax.numpy as jnp
import optax
from jax.flatten_util import ravel_pytree


ArrayTree = Any


def classifier_logits(features: jnp.ndarray, classifier_params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Apply the final linear classifier to frozen feature vectors."""
    logits = features @ classifier_params["kernel"]
    if "bias" in classifier_params:
        logits = logits + classifier_params["bias"]
    return logits


def classifier_loss(
    classifier_params: Dict[str, jnp.ndarray],
    features: jnp.ndarray,
    labels: jnp.ndarray,
) -> jnp.ndarray:
    """Compute mean cross-entropy for the final classifier layer only."""
    logits = classifier_logits(features, classifier_params)
    losses = optax.softmax_cross_entropy_with_integer_labels(logits, labels.astype(jnp.int32))
    return jnp.mean(losses)


def classifier_grad(
    classifier_params: Dict[str, jnp.ndarray],
    features: jnp.ndarray,
    labels: jnp.ndarray,
) -> ArrayTree:
    """Differentiate final-layer loss with respect to final-layer params."""
    return jax.grad(classifier_loss)(classifier_params, features, labels)


def last_layer_grad_per_example(
    features: jnp.ndarray,
    labels: jnp.ndarray,
    classifier_params: Dict[str, jnp.ndarray],
) -> Dict[str, jnp.ndarray]:
    """Return per-example gradients for the top fully connected layer."""
    logits = classifier_logits(features, classifier_params)
    # dCE/dlogits for softmax cross entropy is softmax(logits) - one_hot(y).
    residual = jax.nn.softmax(logits, axis=-1) - jax.nn.one_hot(
        labels.astype(jnp.int32),
        logits.shape[-1],
    )
    grads = {
        # Kernel gradient has shape [batch, feature_dim, num_classes].
        "kernel": jnp.einsum("bd,bc->bdc", features, residual),
    }
    if "bias" in classifier_params:
        grads["bias"] = residual
    return grads


def _per_example_dot(per_example_grads: Dict[str, jnp.ndarray], vector: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Dot each per-example gradient with a fixed vector such as s_test."""
    dots = jnp.einsum("bdc,dc->b", per_example_grads["kernel"], vector["kernel"])
    if "bias" in per_example_grads and "bias" in vector:
        dots = dots + jnp.einsum("bc,c->b", per_example_grads["bias"], vector["bias"])
    return dots


def conjugate_gradient(
    matvec: Callable[[jnp.ndarray], jnp.ndarray],
    b: jnp.ndarray,
    max_iter: int = 50,
    ridge: float = 1e-8,
) -> jnp.ndarray:
    """Solve A x = b using a fixed-length conjugate-gradient loop."""
    x = jnp.zeros_like(b)
    r = b - matvec(x)
    p = r
    rs_old = jnp.vdot(r, r)

    def body(carry, _):
        """Run one CG iteration in a JAX scan-friendly form."""
        x, r, p, rs_old = carry
        ap = matvec(p)
        # ridge avoids division by zero if the search direction degenerates.
        alpha = rs_old / (jnp.vdot(p, ap) + ridge)
        x_new = x + alpha * p
        r_new = r - alpha * ap
        rs_new = jnp.vdot(r_new, r_new)
        beta = rs_new / (rs_old + ridge)
        p_new = r_new + beta * p
        return (x_new, r_new, p_new, rs_new), None

    (x, _, _, _), _ = jax.lax.scan(body, (x, r, p, rs_old), xs=None, length=max_iter)
    return x


def compute_s_test(
    classifier_params: Dict[str, jnp.ndarray],
    train_features: jnp.ndarray,
    train_labels: jnp.ndarray,
    val_features: jnp.ndarray,
    val_labels: jnp.ndarray,
    damping: float = 1e-2,
    cg_iters: int = 50,
) -> Dict[str, jnp.ndarray]:
    """Estimate s_test = (H_train + damping I)^-1 grad L_val."""
    val_grad = classifier_grad(classifier_params, val_features, val_labels)
    flat_val_grad, unravel = ravel_pytree(val_grad)

    def grad_flat(params):
        """Flatten the train loss gradient for HVP computation."""
        grad_tree = classifier_grad(params, train_features, train_labels)
        flat_grad, _ = ravel_pytree(grad_tree)
        return flat_grad

    def hvp_flat(vector_flat):
        """Compute a damped Hessian-vector product in flattened space."""
        vector_tree = unravel(vector_flat)
        _, hvp_tree = jax.jvp(grad_flat, (classifier_params,), (vector_tree,))
        return hvp_tree + damping * vector_flat

    # CG works on the flattened parameter vector, then unravels to a PyTree.
    flat_s_test = conjugate_gradient(hvp_flat, flat_val_grad, max_iter=cg_iters)
    return unravel(flat_s_test)


def s_test_residual_norm(
    classifier_params: Dict[str, jnp.ndarray],
    train_features: jnp.ndarray,
    train_labels: jnp.ndarray,
    val_features: jnp.ndarray,
    val_labels: jnp.ndarray,
    s_test: Dict[str, jnp.ndarray],
    damping: float = 1e-2,
) -> jnp.ndarray:
    """Measure relative residual of the damped iHVP linear solve."""
    val_grad = classifier_grad(classifier_params, val_features, val_labels)
    flat_val_grad, unravel = ravel_pytree(val_grad)
    flat_s_test, _ = ravel_pytree(s_test)

    def grad_flat(params):
        """Flatten train gradients to reuse the same HVP as compute_s_test."""
        grad_tree = classifier_grad(params, train_features, train_labels)
        flat_grad, _ = ravel_pytree(grad_tree)
        return flat_grad

    _, hvp = jax.jvp(grad_flat, (classifier_params,), (unravel(flat_s_test),))
    # Residual is zero when (H + damping I) s_test equals grad L_val.
    residual = hvp + damping * flat_s_test - flat_val_grad
    return jnp.linalg.norm(residual) / (jnp.linalg.norm(flat_val_grad) + 1e-12)


def influence_up_loss(
    features: jnp.ndarray,
    labels: jnp.ndarray,
    classifier_params: Dict[str, jnp.ndarray],
    s_test: Dict[str, jnp.ndarray],
) -> jnp.ndarray:
    """Compute I_up_loss for each example using the last-layer approximation."""
    per_example_grads = last_layer_grad_per_example(features, labels, classifier_params)
    # Negative sign follows the influence-up convention for validation loss.
    return -_per_example_dot(per_example_grads, s_test)
