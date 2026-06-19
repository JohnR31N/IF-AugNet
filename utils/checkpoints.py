from pathlib import Path
import json
import pickle
from typing import TypeVar

from flax import serialization


T = TypeVar("T")


def save_state(path: str, state: T) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(serialization.to_bytes(state))


def _restore_without_opt_state(payload: bytes, state: T) -> T:
    checkpoint_state = serialization.msgpack_restore(payload)
    target_state = serialization.to_state_dict(state)
    for key, value in checkpoint_state.items():
        if key == "opt_state":
            continue
        if key in target_state:
            target_state[key] = value
    return serialization.from_state_dict(state, target_state)


def restore_state(path: str, state: T, restore_opt_state: bool = True) -> T:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    payload = checkpoint_path.read_bytes()
    if not restore_opt_state:
        return _restore_without_opt_state(payload, state)
    try:
        return serialization.from_bytes(state, payload)
    except ValueError as exc:
        if "opt_state" not in str(exc):
            raise

        return _restore_without_opt_state(payload, state)


def save_json(path: str, payload: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_pickle(path: str, payload: object) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as f:
        pickle.dump(payload, f)


def load_pickle(path: str) -> object:
    with Path(path).open("rb") as f:
        return pickle.load(f)
