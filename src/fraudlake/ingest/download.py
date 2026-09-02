"""Kaggle download stage.

Uses the official ``kaggle`` CLI (credentials from ``~/.kaggle/kaggle.json`` or
``KAGGLE_USERNAME``/``KAGGLE_KEY``). The competition requires accepting its
rules once on the website before the API will serve files.
"""

from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path

from rich.console import Console

from fraudlake.config import KAGGLE_COMPETITION, RAW_FILES, Settings

console = Console()


def download_raw(settings: Settings, force: bool = False) -> list[Path]:
    settings.ensure_dirs()
    raw = settings.raw_dir
    present = [raw / f for f in RAW_FILES if (raw / f).exists()]
    if len(present) == len(RAW_FILES) and not force:
        console.print(f"[green]raw files already present in {raw}; skipping download[/]")
        return present

    console.print(f"[cyan]downloading {KAGGLE_COMPETITION} to {raw} ...[/]")
    cmd = ["kaggle", "competitions", "download", "-c", KAGGLE_COMPETITION, "-p", str(raw)]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:  # pragma: no cover
        raise RuntimeError(
            "kaggle CLI not found; run `uv sync` and add ~/.kaggle/kaggle.json"
        ) from exc
    except subprocess.CalledProcessError as exc:  # pragma: no cover
        raise RuntimeError(
            "kaggle download failed. Have you accepted the competition rules at "
            f"https://www.kaggle.com/competitions/{KAGGLE_COMPETITION}/rules ?"
        ) from exc

    for z in raw.glob("*.zip"):
        console.print(f"  extracting {z.name}")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(raw)
        z.unlink()

    missing = [f for f in RAW_FILES if not (raw / f).exists()]
    if missing:
        raise RuntimeError(f"expected files missing after download: {missing}")
    return [raw / f for f in RAW_FILES]
