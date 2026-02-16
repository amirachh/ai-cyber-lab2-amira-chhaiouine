from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_directory(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(data: dict[str, Any], filepath: str | Path) -> None:
    file_path = Path(filepath)
    ensure_directory(file_path.parent)
    file_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def save_plot(plt_object: Any, filepath: str | Path) -> None:
    file_path = Path(filepath)
    ensure_directory(file_path.parent)
    plt_object.savefig(file_path)
