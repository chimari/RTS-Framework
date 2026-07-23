from __future__ import annotations
import hashlib, json, shutil
from pathlib import Path

def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()

def prepare_output_dir(path: Path, overwrite: bool) -> Path:
    path = path.expanduser().resolve()
    if path.exists() and any(path.iterdir()):
        if not overwrite: raise FileExistsError(f"{path} is not empty; use --overwrite")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "_-+" else "_" for c in s)
