"""Integration test for Step 05 corrected FITS output v5.4.0."""

from __future__ import annotations

import csv
import hashlib
import json
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
from steps import step05_apply_rts_correction as step05


def require(condition: bool, message: str) -> None:
    if not condition:
        print(f"FAIL: {message}")
        raise SystemExit(1)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(root: Path) -> Path:
    paths = []
    for index in range(8):
        data = np.array(
            [
                [0 if index % 4 < 2 else 10, 20, 30],
                [10 if index % 4 < 2 else 0, 5, 50],
            ],
            dtype=np.uint16,
        )
        path = root / f"bias_{index:04d}.fit"
        fits.PrimaryHDU(data=data).writeto(path, overwrite=True)
        paths.append(path)

    rows = []
    for index, path in enumerate(paths):
        rows.append({
            "dataset": "bias",
            "directory": str(root),
            "environment": "step05-v5.4-test",
            "frame_index": index,
            "n_frames": len(paths),
            "temperature_C": -10.0,
            "temperature_start_C": -10.0,
            "temperature_end_C": -10.0,
            "temperature_fraction": index / (len(paths) - 1),
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


def prepare_application(
    root: Path,
) -> tuple[step05.RTSCorrectionApplicationResult, Path, np.ndarray]:
    manifest = write_manifest(root)
    analysis = step03.prepare_bias_analysis(manifest, "bias")
    plan04 = step04.prepare_rts_dictionary_analysis(analysis)
    built = step04.build_rts_dictionary_artifacts(
        plan04,
        root / "dictionary.csv",
        minimum_score=0.9,
        minimum_state_count=2,
        minimum_separation=5.0,
        minimum_transition_count=3,
        minimum_lower_run=2,
        minimum_upper_run=2,
    )
    step04.audit_rts_dictionary_input_files(built.metadata_path)

    source = root / "target.fit"
    initial = np.full((2, 3), 20, dtype=np.float32)
    header = fits.Header()
    header["OBJECT"] = "RTS_TEST"
    header["EXPTIME"] = 12.5
    fits.PrimaryHDU(data=initial, header=header).writeto(
        source, overwrite=True
    )

    plan = step05.prepare_rts_correction(built.metadata_path, source)
    values = tuple(
        candidate.lower_state_center + 0.1 * candidate.state_separation
        if index % 2 == 0
        else candidate.upper_state_center - 0.1 * candidate.state_separation
        for index, candidate in enumerate(plan.candidates)
    )
    for candidate, value in zip(plan.candidates, values, strict=True):
        initial[candidate.row, candidate.column] = value
    fits.PrimaryHDU(data=initial, header=header).writeto(
        source, overwrite=True
    )

    classifications = step05.classify_rts_correction_candidates(plan)
    decisions = step05.build_rts_correction_decisions(classifications)
    application = step05.apply_rts_correction_in_memory(decisions)
    return application, source, initial


def main() -> int:
    print("=" * 72)
    print("RTS Framework Step 05 corrected FITS output test")
    print("=" * 72)
    print(f"step05 version : {step05.__version__}")
    print()

    with tempfile.TemporaryDirectory(prefix="rts_step05_write_") as temp:
        root = Path(temp)
        application, source, original = prepare_application(root)
        source_hash = sha256(source)
        output = root / "corrected.fit"

        print("[1/4] A new corrected FITS is written without changing input")
        result = step05.write_rts_corrected_fits(application, output)
        require(output.exists(), "corrected FITS was not created")
        require(result.written, "written flag changed")
        require(result.verified, "verified flag changed")
        require(sha256(source) == source_hash, "input FITS was modified")
        require(result.output_path == output.resolve(), "output path changed")
        print("   Output FITS   : created")
        print("   Input FITS    : unchanged")
        print("   Written       : True")
        print("   Verified      : True")
        print("   Result        : PASS")
        print()

        print("[2/4] Existing files and input overwrite are rejected")
        try:
            step05.write_rts_corrected_fits(application, output)
        except step05.Step05Error as exc:
            require(
                "already exists" in str(exc),
                f"unexpected existing-file error: {exc}",
            )
        else:
            require(False, "existing output was overwritten")

        try:
            step05.write_rts_corrected_fits(
                application, source, overwrite=True
            )
        except step05.Step05Error as exc:
            require(
                "must not overwrite" in str(exc),
                f"unexpected input-overwrite error: {exc}",
            )
        else:
            require(False, "input FITS overwrite was accepted")
        print("   Existing file: rejected by default")
        print("   Input path    : always rejected")
        print("   Result        : PASS")
        print()

        print("[3/4] Header, HISTORY, SHA256, shape, dtype, and pixels agree")
        with fits.open(output, memmap=False, uint=True) as hdul:
            data = hdul[0].data
            header = hdul[0].header
            require(header["OBJECT"] == "RTS_TEST", "OBJECT changed")
            require(header["EXPTIME"] == 12.5, "EXPTIME changed")
            history = header.get("HISTORY", [])
            if isinstance(history, str):
                history = [history]
            history_text = tuple(str(item) for item in history)
            for expected in result.history_entries:
                require(
                    expected in history_text,
                    f"missing HISTORY entry: {expected}",
                )
            require(
                tuple(data.shape) == application.plan.image_shape,
                "written shape changed",
            )
            require(
                data.dtype.name == application.output_dtype,
                "written dtype changed",
            )
            require(
                np.array_equal(
                    data, application.corrected_image, equal_nan=True
                ),
                "written pixels changed",
            )

        require(result.sha256 == sha256(output), "SHA256 changed")
        require(len(result.sha256) == 64, "SHA256 length changed")
        print("   Source header : preserved")
        print("   HISTORY       : appended")
        print("   Shape/dtype   : verified")
        print("   Pixel values  : verified")
        print("   SHA256        : verified")
        print("   Result        : PASS")
        print()

        print("[4/4] Output result is immutable and deterministic")
        overwrite_result = step05.write_rts_corrected_fits(
            application, output, overwrite=True
        )
        require(
            result.summary() == overwrite_result.summary(),
            "output summary changed",
        )
        require(
            json.dumps(result.summary(), sort_keys=True)
            == json.dumps(overwrite_result.summary(), sort_keys=True),
            "serialized output summary changed",
        )
        try:
            result.verified = False
        except (FrozenInstanceError, AttributeError):
            pass
        else:
            require(False, "output result is mutable")
        print("   Output result : immutable")
        print("   Repeated write: deterministic")
        print("   Summary       : deterministic")
        print("   Result        : PASS")
        print()

    print("=" * 72)
    print("FINISHED: Step 05 corrected FITS output test passed")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
