from importlib import metadata
from pathlib import Path
import json
import platform
import sys
from typing import Any, Dict

import jax


def write_run_manifest(path: str, config: Dict[str, Any], command: str) -> None:
    packages = {}
    for name in ("jax", "jaxlib", "flax", "optax", "numpy"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None

    manifest = {
        "command": command,
        "config": config,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": packages,
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(device) for device in jax.devices()],
        },
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
