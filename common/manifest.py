"""
RTS Framework manifest data model.

Design principles
-----------------
- FrameRecord is the fundamental metadata unit.
- This module stores and validates metadata only.
- Image loading and image-file validation belong to ``common.image_io``.
- Analysis algorithms should not depend on CSV columns or image formats.
- Public collections are exposed as immutable tuples.

This first implementation intentionally does not include CSV loading.
``FrameManifest.from_csv()`` and CSV export will be added after the
data model and validation API have been reviewed in actual use.
"""

from __future__ import annotations

__version__ = "2.0.0"

import csv
import math

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Literal, Mapping, overload


IssueSeverity = Literal["error", "warning"]


# ============================================================================
# Exceptions
# ============================================================================


class ManifestError(Exception):
    """Base exception for manifest-related errors."""


class DatasetNotFoundError(ManifestError):
    """Raised when a requested dataset does not exist."""

    def __init__(self, dataset_name: str) -> None:
        self.dataset_name = dataset_name
        super().__init__(f"Dataset not found: {dataset_name!r}")



# ============================================================================
# CSV schema
# ============================================================================


class ManifestSchema:
    """Definition and scalar conversion rules for the manifest CSV format."""

    SCHEMA_VERSION = "1.0"

    REQUIRED_COLUMNS = (
        "dataset",
        "directory",
        "environment",
        "frame_index",
        "n_frames",
        "temperature_C",
        "temperature_start_C",
        "temperature_end_C",
        "temperature_fraction",
        "exposure_s",
        "filename",
        "filepath",
    )

    OPTIONAL_COLUMNS = (
        "image_width",
        "image_height",
        "pixel_dtype",
        "byte_order",
    )

    FIELD_CONVERTERS = {
        "dataset": str,
        "directory": str,
        "environment": str,
        "frame_index": int,
        "n_frames": int,
        "temperature_C": float,
        "temperature_start_C": float,
        "temperature_end_C": float,
        "temperature_fraction": float,
        "exposure_s": float,
        "filename": str,
        "filepath": Path,
        "image_width": int,
        "image_height": int,
        "pixel_dtype": str,
        "byte_order": str,
    }

    @classmethod
    def all_columns(cls) -> tuple[str, ...]:
        """Return all recognized columns in canonical order."""
        return cls.REQUIRED_COLUMNS + cls.OPTIONAL_COLUMNS

    @classmethod
    def validate_columns(cls, fieldnames: Iterable[str] | None) -> None:
        """Validate a CSV header and raise :class:`ManifestError` on failure."""
        if fieldnames is None:
            raise ManifestError("Manifest CSV has no header row.")

        columns = tuple(fieldnames)
        if not columns:
            raise ManifestError("Manifest CSV has an empty header row.")

        blank_positions = [index + 1 for index, name in enumerate(columns) if not name]
        if blank_positions:
            positions = ", ".join(str(value) for value in blank_positions)
            raise ManifestError(f"Manifest CSV has blank column name(s) at position(s): {positions}.")

        duplicates = sorted({name for name in columns if columns.count(name) > 1})
        if duplicates:
            names = ", ".join(repr(name) for name in duplicates)
            raise ManifestError(f"Manifest CSV has duplicate column name(s): {names}.")

        missing = [name for name in cls.REQUIRED_COLUMNS if name not in columns]
        if missing:
            names = ", ".join(repr(name) for name in missing)
            raise ManifestError(f"Manifest CSV is missing required column(s): {names}.")

    @classmethod
    def convert(
        cls,
        field_name: str,
        raw_value: str | None,
        *,
        csv_line: int,
    ) -> object:
        """Convert one CSV scalar to its declared Python representation."""
        if field_name not in cls.FIELD_CONVERTERS:
            raise ManifestError(f"Unknown manifest field: {field_name!r}.")

        is_optional = field_name in cls.OPTIONAL_COLUMNS
        text = "" if raw_value is None else raw_value.strip()

        if text == "":
            if is_optional:
                return None
            raise ManifestError(
                f"CSV line {csv_line}: required field {field_name!r} is empty."
            )

        converter = cls.FIELD_CONVERTERS[field_name]
        try:
            return converter(text)
        except (TypeError, ValueError) as exc:
            expected = {
                int: "an integer",
                float: "a floating-point number",
                Path: "a path",
                str: "text",
            }.get(converter, converter.__name__)
            raise ManifestError(
                f"CSV line {csv_line}: field {field_name!r} must be {expected}; "
                f"got {raw_value!r}."
            ) from exc

# ============================================================================
# Frame and dataset records
# ============================================================================


@dataclass(slots=True, frozen=True)
class FrameRecord:
    """
    Metadata describing one acquired image frame.

    This object contains no image data. It only records where the image is
    located and under what conditions it was acquired.

    Parameters
    ----------
    manifest_row
        Physical line number in the source manifest CSV.

        The header occupies line 1, therefore the first data record
        has ``manifest_row == 2``.
    dataset
        Dataset or acquisition-sequence name.
    directory
        Directory field preserved from the source manifest.
    environment
        Free-form acquisition-environment label.
    frame_index
        Frame index within the dataset.
    n_frames
        Number of frames declared for the dataset.
    temperature_C
        Measured frame temperature in degrees Celsius.
    temperature_start_C, temperature_end_C
        Temperature values recorded at the beginning and end of acquisition.
        These preserve acquisition order and are not required to be sorted.
    temperature_fraction
        Relative position within the temperature sequence, normally 0 to 1.
    exposure_s
        Exposure time in seconds.
    filename
        Image filename.
    filepath
        Resolved or unresolved path to the image file.
    image_width, image_height
        Optional image dimensions in pixels.
    pixel_dtype
        Optional NumPy-compatible pixel dtype string.
    byte_order
        Optional byte-order label, normally ``"little"`` or ``"big"``.
    """

    manifest_row: int

    dataset: str
    directory: str
    environment: str

    frame_index: int
    n_frames: int

    temperature_C: float
    temperature_start_C: float
    temperature_end_C: float
    temperature_fraction: float

    exposure_s: float

    filename: str
    filepath: Path

    image_width: int | None = None
    image_height: int | None = None
    pixel_dtype: str | None = None
    byte_order: str | None = None

    def __post_init__(self) -> None:
        """Normalize ``filepath`` while retaining dataclass immutability."""
        object.__setattr__(self, "filepath", Path(self.filepath))


@dataclass(slots=True, frozen=True)
class DatasetRecord:
    """
    Immutable collection of frames belonging to one acquisition sequence.
    """

    name: str
    frames: tuple[FrameRecord, ...]

    @property
    def n_frames(self) -> int:
        """Return the number of actual frames in this dataset."""
        return len(self.frames)

    @property
    def declared_n_frames(self) -> tuple[int, ...]:
        """Return the distinct ``n_frames`` values declared by its records."""
        return tuple(sorted({frame.n_frames for frame in self.frames}))

    @property
    def temperature_range(self) -> tuple[float, float] | None:
        """Return minimum and maximum measured temperature."""
        if not self.frames:
            return None
        temperatures = [frame.temperature_C for frame in self.frames]
        return min(temperatures), max(temperatures)

    @property
    def exposure_values(self) -> tuple[float, ...]:
        """Return sorted unique exposure times in seconds."""
        return tuple(sorted({frame.exposure_s for frame in self.frames}))

    @property
    def first_frame(self) -> FrameRecord | None:
        """Return the first stored frame, or ``None`` for an empty dataset."""
        return self.frames[0] if self.frames else None

    @property
    def last_frame(self) -> FrameRecord | None:
        """Return the last stored frame, or ``None`` for an empty dataset."""
        return self.frames[-1] if self.frames else None

    def summary(self) -> str:
        """Return a human-readable dataset summary."""
        lines = [
            f"Dataset: {self.name}",
            "=" * (9 + len(self.name)),
            f"Frames       : {self.n_frames}",
        ]

        temperature_range = self.temperature_range
        if temperature_range is None:
            lines.append("Temperature  : n/a")
        else:
            t_min, t_max = temperature_range
            lines.append(f"Temperature  : {t_min:+.3f} ... {t_max:+.3f} °C")

        if self.exposure_values:
            formatted = ", ".join(_format_exposure(value) for value in self.exposure_values)
            lines.append(f"Exposure     : {formatted}")
        else:
            lines.append("Exposure     : n/a")

        return "\n".join(lines)

    def __iter__(self) -> Iterator[FrameRecord]:
        return iter(self.frames)

    def __len__(self) -> int:
        return len(self.frames)

    @overload
    def __getitem__(self, index: int) -> FrameRecord:
        ...

    @overload
    def __getitem__(self, index: slice) -> tuple[FrameRecord, ...]:
        ...

    def __getitem__(
        self,
        index: int | slice,
    ) -> FrameRecord | tuple[FrameRecord, ...]:
        return self.frames[index]


# ============================================================================
# Validation results
# ============================================================================


@dataclass(slots=True, frozen=True)
class ManifestIssue:
    """One structural issue found in a frame manifest."""

    severity: IssueSeverity
    issue_type: str
    detail: str
    manifest_row: int | None = None
    dataset: str | None = None

    def __post_init__(self) -> None:
        if self.severity not in ("error", "warning"):
            raise ValueError(
                f"Unsupported issue severity: {self.severity!r}. "
                "Expected 'error' or 'warning'."
            )

    def format(self) -> str:
        """Return a compact one-line representation."""
        location_parts: list[str] = []

        if self.dataset is not None:
            location_parts.append(f"dataset={self.dataset!r}")

        if self.manifest_row is not None:
            location_parts.append(f"row={self.manifest_row}")

        location = f" ({', '.join(location_parts)})" if location_parts else ""
        return (
            f"{self.severity.upper():7s} "
            f"[{self.issue_type}]{location}: {self.detail}"
        )


@dataclass(slots=True, frozen=True)
class ManifestValidation:
    """Result returned by :meth:`FrameManifest.validate_structure`."""

    issues: tuple[ManifestIssue, ...]

    @property
    def valid(self) -> bool:
        """Return ``True`` when no errors were found."""
        return self.error_count == 0

    @property
    def error_count(self) -> int:
        """Return the number of error-level issues."""
        return sum(issue.severity == "error" for issue in self.issues)

    @property
    def warning_count(self) -> int:
        """Return the number of warning-level issues."""
        return sum(issue.severity == "warning" for issue in self.issues)

    @property
    def errors(self) -> tuple[ManifestIssue, ...]:
        """Return only error-level issues."""
        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def warnings(self) -> tuple[ManifestIssue, ...]:
        """Return only warning-level issues."""
        return tuple(issue for issue in self.issues if issue.severity == "warning")

    def summary(self, *, max_issues: int | None = 20) -> str:
        """
        Return a human-readable validation report.

        Parameters
        ----------
        max_issues
            Maximum number of issues included in the report. Set to ``None``
            to show all issues.
        """
        status = "PASSED" if self.valid else "FAILED"
        lines = [
            "Manifest Validation",
            "===================",
            f"Status   : {status}",
            f"Errors   : {self.error_count}",
            f"Warnings : {self.warning_count}",
        ]

        if not self.issues:
            return "\n".join(lines)

        lines.extend(["", "Issues", "------"])

        shown_issues = (
            self.issues
            if max_issues is None
            else self.issues[: max(0, max_issues)]
        )

        lines.extend(issue.format() for issue in shown_issues)

        omitted = len(self.issues) - len(shown_issues)
        if omitted > 0:
            lines.append(f"... {omitted} additional issue(s) omitted")

        return "\n".join(lines)

    def __bool__(self) -> bool:
        return self.valid


# ============================================================================
# Manifest
# ============================================================================


class FrameManifest:
    """
    Central metadata container of the RTS Framework.

    A ``FrameManifest`` represents one analysis session. It contains immutable
    frame and dataset collections, provides common metadata summaries, and
    validates manifest structure. It does not read image data.

    Parameters
    ----------
    frames
        Iterable of frame records. The iterable is consumed immediately and
        stored as an immutable tuple.
    source_path
        Optional path from which the manifest was loaded.
    frame_root
        Optional root directory used to resolve relative image paths.
    """

    def __init__(
        self,
        frames: Iterable[FrameRecord],
        *,
        source_path: str | Path | None = None,
        frame_root: str | Path | None = None,
    ) -> None:
        
        self._frames = tuple(frames)
    
        self._source_path = (
            Path(source_path)
            if source_path is not None
            else None
        )

        self._frame_root = (
            Path(frame_root)
            if frame_root is not None
            else None
        )

        self._datasets = self._build_datasets()

        self._datasets_by_name = {
            dataset.name: dataset
            for dataset in self._datasets
        }


    @classmethod
    def from_csv(
        cls,
        csv_path: str | Path,
        *,
        frame_root: str | Path | None = None,
        encoding: str = "utf-8-sig",
    ) -> "FrameManifest":
        """Load a manifest CSV without opening any referenced image files."""
        source_path = Path(csv_path)
        resolved_root = Path(frame_root) if frame_root is not None else None
        frames: list[FrameRecord] = []

        try:
            with source_path.open("r", encoding=encoding, newline="") as stream:
                reader = csv.DictReader(stream)
                ManifestSchema.validate_columns(reader.fieldnames)

                for csv_line, row in enumerate(reader, start=2):
                    if None in row:
                        extras = row[None]
                        raise ManifestError(
                            f"CSV line {csv_line}: unexpected extra value(s): {extras!r}."
                        )
                    frames.append(
                        cls._frame_from_row(
                            row,
                            csv_line=csv_line,
                            frame_root=resolved_root,
                        )
                    )
        except ManifestError:
            raise
        except (OSError, UnicodeError, csv.Error) as exc:
            raise ManifestError(
                f"Unable to read manifest CSV {str(source_path)!r}: {exc}"
            ) from exc

        return cls(
            frames,
            source_path=source_path,
            frame_root=resolved_root,
        )

    @staticmethod
    def _frame_from_row(
        row: Mapping[str, str | None],
        *,
        csv_line: int,
        frame_root: Path | None,
    ) -> FrameRecord:
        """Convert one parsed CSV row into a :class:`FrameRecord`."""
        values: dict[str, object] = {"manifest_row": csv_line}

        for field_name in ManifestSchema.REQUIRED_COLUMNS:
            values[field_name] = ManifestSchema.convert(
                field_name,
                row.get(field_name),
                csv_line=csv_line,
            )

        for field_name in ManifestSchema.OPTIONAL_COLUMNS:
            values[field_name] = ManifestSchema.convert(
                field_name,
                row.get(field_name),
                csv_line=csv_line,
            )

        filepath = values["filepath"]
        if not isinstance(filepath, Path):
            raise ManifestError(
                f"CSV line {csv_line}: internal filepath conversion failed."
            )
        values["filepath"] = FrameManifest._resolve_filepath(filepath, frame_root)

        return FrameRecord(**values)  # type: ignore[arg-type]

    @staticmethod
    def _resolve_filepath(filepath: Path, frame_root: Path | None) -> Path:
        """Resolve a relative image path below ``frame_root`` when supplied."""
        if frame_root is None or filepath.is_absolute():
            return filepath
        return frame_root / filepath

    def _build_datasets(self) -> tuple[DatasetRecord, ...]:
        grouped: dict[str, list[FrameRecord]] = defaultdict(list)

        for frame in self._frames:
            grouped[frame.dataset].append(frame)

        # Dataset names are sorted for stable output. Frame order is preserved
        # exactly as supplied by the source manifest.
        return tuple(
            DatasetRecord(name=name, frames=tuple(grouped[name]))
            for name in sorted(grouped)
        )

    @property
    def frames(self) -> tuple[FrameRecord, ...]:
        """Return all frames as an immutable tuple."""
        return self._frames

    @property
    def datasets(self) -> tuple[DatasetRecord, ...]:
        """Return all datasets sorted by dataset name."""
        return self._datasets

    @property
    def dataset_names(self) -> tuple[str, ...]:
        """Return all dataset names in stable sorted order."""
        return tuple(dataset.name for dataset in self._datasets)

    @property
    def n_frames(self) -> int:
        """Return the total number of frames."""
        return len(self._frames)

    @property
    def n_datasets(self) -> int:
        """Return the total number of datasets."""
        return len(self._datasets)

    @property
    def environments(self) -> tuple[str, ...]:
        """Return sorted unique non-empty environment labels."""
        return tuple(
            sorted(
                {
                    frame.environment
                    for frame in self._frames
                    if frame.environment.strip()
                }
            )
        )

    @property
    def exposure_values(self) -> tuple[float, ...]:
        """Return sorted unique exposure times in seconds."""
        return tuple(sorted({frame.exposure_s for frame in self._frames}))

    @property
    def temperature_range(self) -> tuple[float, float] | None:
        """Return minimum and maximum measured temperature."""
        if not self._frames:
            return None

        temperatures = [frame.temperature_C for frame in self._frames]
        return min(temperatures), max(temperatures)

    @property
    def image_geometry(self) -> tuple[int, int] | None:
        """
        Return common ``(width, height)`` when all records define one geometry.

        Returns ``None`` when dimensions are missing or inconsistent.
        """
        if not self._frames:
            return None

        geometries: set[tuple[int, int]] = set()

        for frame in self._frames:
            if frame.image_width is None or frame.image_height is None:
                return None
            geometries.add((frame.image_width, frame.image_height))

        if len(geometries) != 1:
            return None

        return next(iter(geometries))

    @property
    def paths(self) -> tuple[Path, ...]:
        """Return all frame paths in manifest order."""
        return tuple(frame.filepath for frame in self._frames)

    @property
    def source_path(self) -> Path | None:
        """Return the CSV source path, when known."""
        return self._source_path

    @property
    def frame_root(self) -> Path | None:
        """Return the optional root used to resolve relative frame paths."""
        return self._frame_root

    @property
    def is_empty(self) -> bool:
        """Return ``True`` when the manifest has no frame records."""
        return not self._frames
    
    def get_dataset(self, name: str) -> DatasetRecord:
        """Return one dataset by exact name."""
        try:
            return self._datasets_by_name[name]
        except KeyError as exc:
            raise DatasetNotFoundError(name) from exc

    def summary(self) -> str:
        """Return a human-readable manifest summary."""
        lines = [
            "Frame Manifest",
            "==============",
            f"Frames       : {self.n_frames}",
            f"Datasets     : {self.n_datasets}",
        ]

        temperature_range = self.temperature_range
        if temperature_range is None:
            lines.append("Temperature  : n/a")
        else:
            t_min, t_max = temperature_range
            lines.append(f"Temperature  : {t_min:+.3f} ... {t_max:+.3f} °C")

        if self.exposure_values:
            exposure_text = ", ".join(
                _format_exposure(value) for value in self.exposure_values
            )
            lines.append(f"Exposure     : {exposure_text}")
        else:
            lines.append("Exposure     : n/a")

        geometry = self.image_geometry
        if geometry is None:
            lines.append("Geometry     : unspecified or inconsistent")
        else:
            width, height = geometry
            lines.append(f"Geometry     : {width} × {height} pixels")

        if self.environments:
            lines.append(f"Environments : {', '.join(self.environments)}")
        else:
            lines.append("Environments : n/a")

        if self.source_path is not None:
            lines.append(f"Source       : {self.source_path}")

        if self.frame_root is not None:
            lines.append(f"Frame root   : {self.frame_root}")

        if self._datasets:
            lines.extend(["", "Datasets", "--------"])
            name_width = max(len(dataset.name) for dataset in self._datasets)

            for dataset in self._datasets:
                temp_range = dataset.temperature_range
                if temp_range is None:
                    temp_text = "temperature n/a"
                else:
                    t_min, t_max = temp_range
                    temp_text = f"{t_min:+.3f} ... {t_max:+.3f} °C"

                lines.append(
                    f"{dataset.name:<{name_width}}  "
                    f"{dataset.n_frames:>6d} frames  "
                    f"{temp_text}"
                )

        return "\n".join(lines)

    def validate_structure(self) -> ManifestValidation:
        """
        Validate metadata consistency without opening image files.

        The validation intentionally reports all discovered issues rather than
        raising on the first problem. Image existence, image shape, and image
        dtype are outside this method's responsibility.
        """
        issues: list[ManifestIssue] = []

        if not self._frames:
            issues.append(
                ManifestIssue(
                    severity="warning",
                    issue_type="empty_manifest",
                    detail="The manifest contains no frame records.",
                )
            )
            return ManifestValidation(tuple(issues))

        issues.extend(self._validate_record_values())
        issues.extend(self._validate_duplicate_paths())
        issues.extend(self._validate_datasets())

        return ManifestValidation(tuple(issues))

    def _validate_record_values(self) -> list[ManifestIssue]:
        issues: list[ManifestIssue] = []

        for frame in self._frames:
            row = frame.manifest_row
            dataset = frame.dataset or None

            if row < 2:
                issues.append(
                    ManifestIssue(
                        "error",
                        "invalid_manifest_row",
                        (
                            "manifest_row must be a physical CSV data-line number "
                            f"greater than or equal to 2, got {row}."
                        ),
                        row,
                        dataset,
                    )
                )

            for field_name, value in (
                ("dataset", frame.dataset),
                ("directory", frame.directory),
                ("filename", frame.filename),
            ):
                if not value.strip():
                    issues.append(
                        ManifestIssue(
                            "error",
                            "empty_required_field",
                            f"{field_name} must not be empty.",
                            row,
                            dataset,
                        )
                    )

            if frame.environment == "":
                issues.append(
                    ManifestIssue(
                        "warning",
                        "empty_environment",
                        "environment is empty.",
                        row,
                        dataset,
                    )
                )

            if frame.frame_index < 0:
                issues.append(
                    ManifestIssue(
                        "error",
                        "negative_frame_index",
                        f"frame_index must be non-negative, got {frame.frame_index}.",
                        row,
                        dataset,
                    )
                )

            if frame.n_frames <= 0:
                issues.append(
                    ManifestIssue(
                        "error",
                        "invalid_declared_frame_count",
                        f"n_frames must be positive, got {frame.n_frames}.",
                        row,
                        dataset,
                    )
                )

            if not 0.0 <= frame.temperature_fraction <= 1.0:
                issues.append(
                    ManifestIssue(
                        "warning",
                        "temperature_fraction_out_of_range",
                        (
                            "temperature_fraction is normally expected within "
                            f"[0, 1], got {frame.temperature_fraction}."
                        ),
                        row,
                        dataset,
                    )
                )

            if not math.isfinite(frame.exposure_s):
                issues.append(
                    ManifestIssue(
                        "error",
                        "invalid_exposure",
                        f"exposure_s must be finite, got {frame.exposure_s}.",
                        row,
                        dataset,
                    )
                )
            
                
            elif frame.exposure_s < 0.0:
                issues.append(
                    ManifestIssue(
                        "error",
                        "invalid_exposure",
                        f"exposure_s must be zero or positive, got {frame.exposure_s}.",
                        row,
                        dataset,
                    )
                )
            

            if frame.filename and frame.filepath.name != frame.filename:
                issues.append(
                    ManifestIssue(
                        "warning",
                        "filename_path_mismatch",
                        (
                            f"filename={frame.filename!r}, but filepath.name="
                            f"{frame.filepath.name!r}."
                        ),
                        row,
                        dataset,
                    )
                )

            if frame.image_width is not None and frame.image_width <= 0:
                issues.append(
                    ManifestIssue(
                        "error",
                        "invalid_image_width",
                        f"image_width must be positive, got {frame.image_width}.",
                        row,
                        dataset,
                    )
                )

            if frame.image_height is not None and frame.image_height <= 0:
                issues.append(
                    ManifestIssue(
                        "error",
                        "invalid_image_height",
                        f"image_height must be positive, got {frame.image_height}.",
                        row,
                        dataset,
                    )
                )

            if (frame.image_width is None) != (frame.image_height is None):
                issues.append(
                    ManifestIssue(
                        "warning",
                        "partial_image_geometry",
                        (
                            "image_width and image_height should either both be "
                            "defined or both be None."
                        ),
                        row,
                        dataset,
                    )
                )

            if (
                frame.byte_order is not None
                and frame.byte_order not in {"little", "big", "native", "not-applicable"}
            ):
                issues.append(
                    ManifestIssue(
                        "warning",
                        "unknown_byte_order",
                        f"Unrecognized byte_order value: {frame.byte_order!r}.",
                        row,
                        dataset,
                    )
                )

        return issues

    def _validate_duplicate_paths(self) -> list[ManifestIssue]:
        issues: list[ManifestIssue] = []
        path_rows: dict[Path, list[FrameRecord]] = defaultdict(list)

        for frame in self._frames:
            path_rows[frame.filepath].append(frame)

        for path, records in path_rows.items():
            if len(records) <= 1:
                continue

            rows = ", ".join(str(record.manifest_row) for record in records)
            issues.append(
                ManifestIssue(
                    "error",
                    "duplicate_filepath",
                    f"filepath {str(path)!r} appears in manifest rows: {rows}.",
                )
            )

        return issues

    def _validate_datasets(self) -> list[ManifestIssue]:
        issues: list[ManifestIssue] = []

        for dataset in self._datasets:
            issues.extend(self._validate_one_dataset(dataset))

        return issues

    def _validate_one_dataset(
        self,
        dataset: DatasetRecord,
    ) -> list[ManifestIssue]:
        issues: list[ManifestIssue] = []
        frames = dataset.frames

        if not frames:
            issues.append(
                ManifestIssue(
                    "warning",
                    "empty_dataset",
                    "The dataset contains no frames.",
                    dataset=dataset.name,
                )
            )
            return issues

        declared_counts = dataset.declared_n_frames
        if len(declared_counts) != 1:
            issues.append(
                ManifestIssue(
                    "error",
                    "inconsistent_declared_frame_count",
                    f"Multiple n_frames values are present: {declared_counts}.",
                    dataset=dataset.name,
                )
            )
        else:
            declared_count = declared_counts[0]
            if declared_count != len(frames):
                issues.append(
                    ManifestIssue(
                        "error",
                        "declared_frame_count_mismatch",
                        (
                            f"n_frames declares {declared_count}, but the dataset "
                            f"contains {len(frames)} records."
                        ),
                        dataset=dataset.name,
                    )
                )

        frame_index_rows: dict[int, list[int]] = defaultdict(list)
        for frame in frames:
            frame_index_rows[frame.frame_index].append(frame.manifest_row)

        duplicate_indices = {
            index: rows
            for index, rows in frame_index_rows.items()
            if len(rows) > 1
        }

        for frame_index, rows in sorted(duplicate_indices.items()):
            rows_text = ", ".join(str(row) for row in rows)
            issues.append(
                ManifestIssue(
                    "error",
                    "duplicate_frame_index",
                    (
                        f"frame_index={frame_index} appears in manifest rows "
                        f"{rows_text}."
                    ),
                    dataset=dataset.name,
                )
            )

        indices_in_manifest_order = [frame.frame_index for frame in frames]
        if any(
            current < previous
            for previous, current in zip(
                indices_in_manifest_order,
                indices_in_manifest_order[1:],
            )
        ):
            issues.append(
                ManifestIssue(
                    "warning",
                    "non_monotonic_frame_index",
                    "frame_index is not monotonically increasing in manifest order.",
                    dataset=dataset.name,
                )
            )

        unique_indices = sorted(frame_index_rows)
        if unique_indices:
            zero_based = list(range(0, len(frames)))
            one_based = list(range(1, len(frames) + 1))

            if unique_indices not in (zero_based, one_based):
                issues.append(
                    ManifestIssue(
                        "warning",
                        "non_contiguous_frame_index",
                        (
                            "frame_index values are not a contiguous zero-based "
                            "or one-based sequence."
                        ),
                        dataset=dataset.name,
                    )
                )

        directory_values = {frame.directory for frame in frames}
        if len(directory_values) > 1:
            issues.append(
                ManifestIssue(
                    "warning",
                    "inconsistent_dataset_directory",
                    (
                        "Multiple directory values are present: "
                        f"{tuple(sorted(directory_values))}."
                    ),
                    dataset=dataset.name,
                )
            )

        geometry_values = {
            (frame.image_width, frame.image_height)
            for frame in frames
            if frame.image_width is not None and frame.image_height is not None
        }
        if len(geometry_values) > 1:
            issues.append(
                ManifestIssue(
                    "error",
                    "inconsistent_image_geometry",
                    (
                        "Multiple image geometries are present: "
                        f"{tuple(sorted(geometry_values))}."
                    ),
                    dataset=dataset.name,
                )
            )

        exposure_values = dataset.exposure_values
        if len(exposure_values) > 1:
            issues.append(
                ManifestIssue(
                    "warning",
                    "multiple_exposure_values",
                    f"Multiple exposure times are present: {exposure_values}.",
                    dataset=dataset.name,
                )
            )

        return issues

    def __iter__(self) -> Iterator[FrameRecord]:
        return iter(self._frames)

    def __len__(self) -> int:
        return len(self._frames)

    @overload
    def __getitem__(self, index: int) -> FrameRecord:
        ...

    @overload
    def __getitem__(self, index: slice) -> tuple[FrameRecord, ...]:
        ...

    def __getitem__(
        self,
        index: int | slice,
    ) -> FrameRecord | tuple[FrameRecord, ...]:
        return self._frames[index]

    def __contains__(self, dataset_name: object) -> bool:
        """Return whether an exact dataset name is present."""
        return isinstance(dataset_name, str) and dataset_name in self._datasets_by_name

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"n_frames={self.n_frames}, "
            f"n_datasets={self.n_datasets})"
        )


# ============================================================================
# Formatting helpers
# ============================================================================


def _format_exposure(exposure_s: float) -> str:
    """Format exposure time using a readable SI unit."""
    absolute = abs(exposure_s)

    if absolute < 1.0e-6:
        return f"{exposure_s * 1.0e9:g} ns"
    if absolute < 1.0e-3:
        return f"{exposure_s * 1.0e6:g} µs"
    if absolute < 1.0:
        return f"{exposure_s * 1.0e3:g} ms"
    return f"{exposure_s:g} s"
