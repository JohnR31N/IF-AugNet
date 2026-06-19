from typing import Tuple

import jax
import jax.numpy as jnp


def average_pool_same(x: jnp.ndarray, kernel_size: int = 4) -> jnp.ndarray:
    """Smooth dense transform fields with same-size average pooling."""
    if kernel_size <= 1:
        return x
    # Edge padding avoids shrinking the field and keeps border flow stable.
    pad_before = (kernel_size - 1) // 2
    pad_after = kernel_size // 2
    padded = jnp.pad(
        x,
        ((0, 0), (pad_before, pad_after), (pad_before, pad_after), (0, 0)),
        mode="edge",
    )
    pooled = jax.lax.reduce_window(
        padded,
        init_value=0.0,
        computation=jax.lax.add,
        window_dimensions=(1, kernel_size, kernel_size, 1),
        window_strides=(1, 1, 1, 1),
        padding="VALID",
    )
    return pooled / float(kernel_size * kernel_size)


def _base_grid(height: int, width: int) -> jnp.ndarray:
    """Create a normalized [-1, 1] grid in y/x order."""
    ys = jnp.linspace(-1.0, 1.0, height)
    xs = jnp.linspace(-1.0, 1.0, width)
    yy, xx = jnp.meshgrid(ys, xs, indexing="ij")
    return jnp.stack([yy, xx], axis=-1)


def _to_pixel_coordinates(grid: jnp.ndarray, height: int, width: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Convert normalized grid coordinates to floating pixel coordinates."""
    y = (grid[..., 0] + 1.0) * 0.5 * (height - 1)
    x = (grid[..., 1] + 1.0) * 0.5 * (width - 1)
    return y, x


def _sample_single(image: jnp.ndarray, grid: jnp.ndarray) -> jnp.ndarray:
    """Sample one image at a dense grid using bilinear interpolation."""
    height, width, _ = image.shape
    y, x = _to_pixel_coordinates(grid, height, width)

    # Neighbor pixels for bilinear interpolation.
    y0 = jnp.floor(y).astype(jnp.int32)
    x0 = jnp.floor(x).astype(jnp.int32)
    y1 = y0 + 1
    x1 = x0 + 1

    y0c = jnp.clip(y0, 0, height - 1)
    x0c = jnp.clip(x0, 0, width - 1)
    y1c = jnp.clip(y1, 0, height - 1)
    x1c = jnp.clip(x1, 0, width - 1)

    # Bilinear weights for top-left, top-right, bottom-left, bottom-right.
    wa = (y1.astype(jnp.float32) - y) * (x1.astype(jnp.float32) - x)
    wb = (y1.astype(jnp.float32) - y) * (x - x0.astype(jnp.float32))
    wc = (y - y0.astype(jnp.float32)) * (x1.astype(jnp.float32) - x)
    wd = (y - y0.astype(jnp.float32)) * (x - x0.astype(jnp.float32))

    Ia = image[y0c, x0c]
    Ib = image[y0c, x1c]
    Ic = image[y1c, x0c]
    Id = image[y1c, x1c]

    sampled = (
        Ia * wa[..., None]
        + Ib * wb[..., None]
        + Ic * wc[..., None]
        + Id * wd[..., None]
    )

    # Coordinates outside the image are filled with black pixels.
    in_bounds = (y >= 0) & (y <= height - 1) & (x >= 0) & (x <= width - 1)
    return jnp.where(in_bounds[..., None], sampled, 0.0)


def bilinear_sample(images: jnp.ndarray, grid: jnp.ndarray) -> jnp.ndarray:
    """Vectorize bilinear sampling over a batch of images."""
    return jax.vmap(_sample_single)(images, grid)


def apply_spatial_transform(
    images: jnp.ndarray,
    spatial_params: jnp.ndarray,
    spatial_scale: float = 0.20,
    smoothing_kernel: int = 4,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Apply the paper-style local affine spatial transform."""
    height, width = images.shape[1], images.shape[2]
    spatial_params = jnp.nan_to_num(spatial_params, nan=0.0, posinf=1.0, neginf=-1.0)
    params = jnp.tanh(average_pool_same(spatial_params, smoothing_kernel))
    # Four channels form a 2x2 local affine matrix; two channels form bias.
    weights = params[..., :4].reshape(params.shape[0], height, width, 2, 2)
    bias = params[..., 4:6]

    base = _base_grid(height, width)
    base_batch = jnp.broadcast_to(base, (images.shape[0], height, width, 2))
    # Delta is a per-pixel affine displacement around the identity grid.
    delta = jnp.einsum("bhwij,bhwj->bhwi", weights, base_batch) + bias
    sample_grid = jnp.clip(base_batch + spatial_scale * delta, -1.5, 1.5)
    sample_grid = jnp.nan_to_num(sample_grid, nan=0.0, posinf=1.5, neginf=-1.5)
    return bilinear_sample(images, sample_grid), sample_grid


def apply_appearance_transform(
    images: jnp.ndarray,
    appearance_params: jnp.ndarray,
    appearance_scale: float = 0.25,
    smoothing_kernel: int = 4,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Apply local appearance/color transforms to an image batch."""
    channels = images.shape[-1]
    appearance_params = jnp.nan_to_num(appearance_params, nan=0.0, posinf=1.0, neginf=-1.0)
    appearance_params = average_pool_same(appearance_params, smoothing_kernel)
    # Each pixel receives a CxC color matrix and C additive bias values.
    weights = appearance_params[..., : channels * channels]
    bias = appearance_params[..., channels * channels :]

    weights = jnp.tanh(weights.reshape(*images.shape[:3], channels, channels))
    bias = jnp.tanh(bias)
    # Local color delta is scaled and clipped to keep images in [0, 1].
    delta = jnp.einsum("bhwij,bhwj->bhwi", weights, images) + bias
    transformed = jnp.clip(images + appearance_scale * delta, 0.0, 1.0)
    return transformed, delta
