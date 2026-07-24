"""Integration test for Step 04 input-file fingerprints v4.27.0."""

from __future__ import annotations

import csv
import hashlib
import sys
import tempfile
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
from astropy.io import fits

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from steps import step03_prepare_bias_analysis as step03
from steps import step04_prepare_rts_dictionary_analysis as step04


def require(condition: bool, message: str) -> None:
    if not condition:
        print(f"FAIL: {message}")
        raise SystemExit(1)


def write_dataset(root: Path) -> Path:
    paths = []
    for frame_index in range(8):
        data = np.array(
            [
                [0 if frame_index % 4 < 2 else 10, 20, 30],
                [0 if frame_index % 2 == 0 else 10, 5, 50],
            ],
            dtype=np.uint16,
        )
        path = root / f"bias_{frame_index:04d}.fit"
        fits.PrimaryHDU(data=data).writeto(path, overwrite=True)
        paths.append(path)

    rows = []
    for frame_index, path in enumerate(paths):
        rows.append({
            "dataset": "bias",
            "directory": str(root),
            "environment": "step04-v4.27-test",
            "frame_index": frame_index,
            "n_frames": len(paths),
            "temperature_C": -10.0,
            "temperature_start_C": -10.0,
            "temperature_end_C": -10.0,
            "temperature_fraction": frame_index / (len(paths) - 1),
            "exposure_s": 0.0,
            "filename": path.name,
            "filepath": str(path),
            "image_width": 3,
            "image_height": 2,
            "pixel_dtype": "uint16",
            "byte_order": "not-applicable",
        })

    manifest = root / "manifest.normalized.csv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def kwargs() -> dict[str, object]:
    return {
        "minimum_score": 0.9,
        "minimum_state_count": 2,
        "minimum_separation": 5.0,
        "minimum_transition_count": 3,
        "minimum_lower_run": 2,
        "minimum_upper_run": 2,
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    print("=" * 72)
    print("RTS Framework Step 04 input-file fingerprint test")
    print("=" * 72)
    print(f"step04 version : {step04.__version__}")
    print()

    with tempfile.TemporaryDirectory(prefix="rts_step04_hashes_") as temp:
        root = Path(temp)
        plan = step04.prepare_rts_dictionary_analysis(
            step03.prepare_bias_analysis(write_dataset(root), "bias")
        )
        built = step04.build_rts_dictionary_artifacts(
            plan, root / "dictionary.csv", **kwargs()
        )

        print("[1/4] Metadata path produces ordered SHA-256 fingerprints")
        result = step04.fingerprint_rts_dictionary_input_files(
            built.metadata_path
        )
        require(
            isinstance(result, step04.RTSInputFileFingerprintSet),
            "wrong fingerprint-set type",
        )
        require(result.algorithm == "sha256", "wrong algorithm")
        require(result.file_count == plan.n_frames, "wrong file count")
        require(
            all(item.index == index
                for index, item in enumerate(result.files)),
            "fingerprint order changed",
        )
        print("   Type       : RTSInputFileFingerprintSet")
        print("   Algorithm  : sha256")
        print("   File count : matched")
        print("   Order      : preserved")
        print("   Result     : PASS")
        print()

        print("[2/4] Sizes, digests, summaries, and immutability are correct")
        metadata = step04.load_rts_dictionary_metadata_json(
            built.metadata_path
        )
        from_object = step04.fingerprint_rts_dictionary_input_files(metadata)
        require(result.summary() == from_object.summary(),
                "path and object results differ")
        for item in result.files:
            require(item.size_bytes == item.path.stat().st_size,
                    "wrong file size")
            require(item.sha256 == sha256(item.path),
                    "wrong SHA-256 digest")
            require(len(item.sha256) == 64, "wrong digest length")
        require(
            result.total_size_bytes
            == sum(item.size_bytes for item in result.files),
            "wrong total size",
        )
        try:
            result.files[0].sha256 = "0" * 64
        except (FrozenInstanceError, AttributeError):
            pass
        else:
            require(False, "fingerprint object is mutable")
        print("   Sizes      : matched")
        print("   Digests    : matched")
        print("   Summary    : deterministic")
        print("   Dataclass  : frozen")
        print("   Result     : PASS")
        print()

        print("[3/4] Repeated fingerprinting is deterministic")
        repeated = step04.fingerprint_rts_dictionary_input_files(
            built.metadata_path
        )
        require(result == repeated, "fingerprint result changed")
        require(result.summary() == repeated.summary(),
                "fingerprint summary changed")
        print("   Result object : identical")
        print("   Summary       : identical")
        print("   Result        : PASS")
        print()

        print("[4/4] File modification changes only the affected fingerprint")
        before = result
        changed_path = before.files[0].path
        original = changed_path.read_bytes()
        changed_path.write_bytes(original + b"\x00")
        after = step04.fingerprint_rts_dictionary_input_files(
            built.metadata_path
        )
        require(
            after.files[0].size_bytes == before.files[0].size_bytes + 1,
            "modified file size did not change",
        )
        require(
            after.files[0].sha256 != before.files[0].sha256,
            "modified file digest did not change",
        )
        require(
            after.files[1:] == before.files[1:],
            "unmodified fingerprints changed",
        )
        print("   Modified size   : changed")
        print("   Modified digest : changed")
        print("   Other files     : unchanged")
        print("   Result          : PASS")
        print()

    print("=" * 72)
    print("FINISHED: Step 04 input-file fingerprint test passed")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
