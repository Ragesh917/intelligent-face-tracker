import json
from pathlib import Path


def load_config(config_path="config.json"):
    """Load project configuration from a JSON file."""
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8-sig") as file:
        config = json.load(file)

    return config