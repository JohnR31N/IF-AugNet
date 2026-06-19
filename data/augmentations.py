import jax
import jax.numpy as jnp


def random_crop_flip(
    images: jnp.ndarray,
    rng: jax.Array,
    padding: int = 4,
    flip_probability: float = 0.5,
) -> jnp.ndarray:
    if padding > 0:
        images = jnp.pad(
            images,
            ((0, 0), (padding, padding), (padding, padding), (0, 0)),
            mode="reflect",
        )

    batch, height, width, channels = images.shape
    crop_height = height - 2 * padding
    crop_width = width - 2 * padding
    rng_crop, rng_flip = jax.random.split(rng)
    offsets = jax.random.randint(rng_crop, (batch, 2), 0, 2 * padding + 1)

    def crop_one(image, offset):
        y, x = offset
        return jax.lax.dynamic_slice(image, (y, x, 0), (crop_height, crop_width, channels))

    cropped = jax.vmap(crop_one)(images, offsets)
    flips = jax.random.bernoulli(rng_flip, flip_probability, (batch, 1, 1, 1))
    return jnp.where(flips, cropped[:, :, ::-1, :], cropped)


def cutout(images: jnp.ndarray, rng: jax.Array, size: int = 16) -> jnp.ndarray:
    batch, height, width, _ = images.shape
    rng_y, rng_x = jax.random.split(rng)
    centers_y = jax.random.randint(rng_y, (batch,), 0, height)
    centers_x = jax.random.randint(rng_x, (batch,), 0, width)
    yy = jnp.arange(height)[None, :, None]
    xx = jnp.arange(width)[None, None, :]

    half = size // 2
    keep_y = (yy < (centers_y[:, None, None] - half)) | (yy >= (centers_y[:, None, None] + half))
    keep_x = (xx < (centers_x[:, None, None] - half)) | (xx >= (centers_x[:, None, None] + half))
    mask = (keep_y | keep_x)[..., None]
    return images * mask.astype(images.dtype)
