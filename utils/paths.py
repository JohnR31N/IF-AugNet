from pathlib import Path


def output_dir_from_arg(value: str, flag: str = "--output-dir") -> Path:
    output_dir = Path(value)
    if output_dir.suffix:
        raise ValueError(
            f"{flag} expects a directory path, not a file path with a suffix: "
            f"{value}"
        )
    return output_dir
