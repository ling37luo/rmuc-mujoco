"""Verified access to the official RMUC reference STEP files.

The STEP is downloaded from RoboMaster's own server and is never mirrored by
this project.  Keeping this logic explicit avoids accidentally treating a
public download as permission to redistribute it.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import tempfile
from typing import BinaryIO, Callable
from urllib.request import Request, urlopen

from .seasons import RMUC2026, FieldSource, SurfaceGuideSource, season_of


# 1.x module constants: the RMUC 2026 pins, kept for existing imports.
_RMUC2026_STEP = RMUC2026.default_source
_RMUC2026_GUIDE = RMUC2026.surface_guide
_RMUC2026_REVIEWED = RMUC2026.last_reviewed_rulebook
OFFICIAL_STEP_URL = _RMUC2026_STEP.step.url
OFFICIAL_STEP_SIZE = _RMUC2026_STEP.step.size_bytes
OFFICIAL_STEP_SHA256 = _RMUC2026_STEP.step.sha256
OFFICIAL_STEP_PRODUCT = _RMUC2026_STEP.step_product
OFFICIAL_SOURCE_PAGE = _RMUC2026_STEP.source_page
OFFICIAL_RULE_CENTRE = RMUC2026.rule_centre
OFFICIAL_RULEBOOK_V2_0_0_URL = _RMUC2026_GUIDE.file.url
OFFICIAL_RULEBOOK_V2_0_0_SIZE = _RMUC2026_GUIDE.file.size_bytes
OFFICIAL_RULEBOOK_V2_0_0_SHA256 = _RMUC2026_GUIDE.file.sha256
LAST_REVIEWED_RULEBOOK_VERSION = _RMUC2026_REVIEWED.version
LAST_REVIEWED_RULEBOOK_PUBLICATION_DATE = _RMUC2026_REVIEWED.publication_date
LAST_REVIEWED_RULEBOOK_URL = _RMUC2026_REVIEWED.file.url
LAST_REVIEWED_RULEBOOK_SIZE = _RMUC2026_REVIEWED.file.size_bytes
LAST_REVIEWED_RULEBOOK_SHA256 = _RMUC2026_REVIEWED.file.sha256
LAST_RULE_REVIEW_DATE = _RMUC2026_REVIEWED.review_date


class DownloadError(RuntimeError):
    """Raised when the official source cannot be fetched or verified."""


@dataclass(frozen=True)
class DownloadedStep:
    path: Path
    size_bytes: int
    sha256: str
    reused: bool


@dataclass(frozen=True)
class DownloadedRulebook:
    """Identity of a pinned local rulebook reference."""

    path: Path
    size_bytes: int
    sha256: str
    reused: bool


def _sha256(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_official_step(path: Path, source: FieldSource = _RMUC2026_STEP) -> DownloadedStep:
    """Verify the immutable size, header identity, and SHA-256 of the STEP."""

    requested = Path(path).expanduser()
    if requested.is_symlink():
        raise DownloadError(f"STEP path must not be a symlink: {requested}")
    candidate = requested.resolve()
    if not candidate.is_file():
        raise DownloadError(f"not a regular STEP file: {candidate}")
    size = candidate.stat().st_size
    expected = source.step
    if size != expected.size_bytes:
        raise DownloadError(f"STEP size mismatch: expected={expected.size_bytes}, actual={size}")
    with candidate.open("rb") as handle:
        header = handle.read(64 * 1024).decode("latin-1", errors="replace")
    if "ISO-10303-21" not in header or source.step_product not in header:
        title = season_of(source).title
        raise DownloadError(f"STEP header does not identify the pinned {title} field assembly")
    digest = _sha256(candidate)
    if digest != expected.sha256:
        raise DownloadError(f"STEP SHA-256 mismatch: expected={expected.sha256}, actual={digest}")
    return DownloadedStep(candidate, size, digest, reused=True)


def _copy_and_hash(
    source: BinaryIO,
    destination: BinaryIO,
    *,
    expected_size: int,
    progress: Callable[[int, int], None] | None,
    chunk_size: int = 8 * 1024 * 1024,
) -> tuple[int, str]:
    digest = hashlib.sha256()
    total = 0
    while True:
        chunk = source.read(chunk_size)
        if not chunk:
            break
        destination.write(chunk)
        digest.update(chunk)
        total += len(chunk)
        if progress is not None:
            progress(total, expected_size)
    destination.flush()
    os.fsync(destination.fileno())
    return total, digest.hexdigest()


def download_official_step(
    destination: Path,
    *,
    acknowledge_reference_only: bool,
    progress: Callable[[int, int], None] | None = None,
    timeout_seconds: float = 60.0,
    source: FieldSource = _RMUC2026_STEP,
) -> DownloadedStep:
    """Download the pinned STEP directly from RoboMaster and verify it.

    ``acknowledge_reference_only`` must be explicit because the publisher calls
    the model a reference and this project's review found no explicit standard
    redistribution grant. This is a conservative distribution policy, not a
    legal determination. This function downloads from the official URL; it
    never uses a project mirror or silently accepts a different file.
    """

    if not acknowledge_reference_only:
        raise DownloadError(
            "explicit acknowledgement required: the official STEP is reference-only "
            f"material; review {source.source_page}"
        )
    requested = Path(destination).expanduser()
    if requested.is_symlink():
        raise DownloadError(f"destination must not be a symlink: {requested}")
    target = requested.resolve()
    if target.exists():
        verified = verify_official_step(target, source)
        return DownloadedStep(
            verified.path,
            verified.size_bytes,
            verified.sha256,
            reused=True,
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.parent.is_symlink():
        raise DownloadError(f"destination parent must not be a symlink: {target.parent}")

    request = Request(source.step.url, headers={"User-Agent": "rmuc-mujoco/0.2"})
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{target.name}.",
            suffix=".part",
            dir=target.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
                size, digest = _copy_and_hash(
                    response,
                    temporary,
                    expected_size=source.step.size_bytes,
                    progress=progress,
                )
        if size != source.step.size_bytes:
            raise DownloadError(
                f"downloaded STEP size mismatch: expected={source.step.size_bytes}, actual={size}"
            )
        if digest != source.step.sha256:
            raise DownloadError(
                f"downloaded STEP SHA-256 mismatch: expected={source.step.sha256}, actual={digest}"
            )
        assert temporary_path is not None
        if target.exists():
            raise DownloadError(
                f"destination appeared during download; refusing overwrite: {target}"
            )
        os.replace(temporary_path, target)
        temporary_path = None
        verified = verify_official_step(target, source)
        return DownloadedStep(
            verified.path,
            verified.size_bytes,
            verified.sha256,
            reused=False,
        )
    except DownloadError:
        raise
    except Exception as exc:
        raise DownloadError(f"official STEP download failed: {exc}") from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _rulebook_identity(guide: SurfaceGuideSource | None) -> tuple[str, int, str]:
    """URL, size and SHA-256 of ``guide``'s PDF; None is the RMUC 2026 V2.0.0."""

    if guide is None:
        return (
            OFFICIAL_RULEBOOK_V2_0_0_URL,
            OFFICIAL_RULEBOOK_V2_0_0_SIZE,
            OFFICIAL_RULEBOOK_V2_0_0_SHA256,
        )
    return guide.file.url, guide.file.size_bytes, guide.file.sha256


def verify_official_rulebook_v2_0_0(
    path: Path, guide: SurfaceGuideSource | None = None
) -> DownloadedRulebook:
    """Verify the exact PDF used by a season's optional overhead guide.

    The default is the RMUC 2026 V2.0.0 rulebook.
    """

    _url, expected_size, expected_sha256 = _rulebook_identity(guide)
    requested = Path(path).expanduser()
    if requested.is_symlink():
        raise DownloadError(f"rulebook path must not be a symlink: {requested}")
    candidate = requested.resolve()
    if not candidate.is_file():
        raise DownloadError(f"not a regular rulebook PDF: {candidate}")
    size = candidate.stat().st_size
    if size != expected_size:
        raise DownloadError(f"rulebook size mismatch: expected={expected_size}, actual={size}")
    with candidate.open("rb") as handle:
        if handle.read(5) != b"%PDF-":
            raise DownloadError("rulebook file does not have a PDF header")
    digest = _sha256(candidate)
    if digest != expected_sha256:
        raise DownloadError(
            f"rulebook SHA-256 mismatch: expected={expected_sha256}, actual={digest}"
        )
    return DownloadedRulebook(candidate, size, digest, reused=True)


def download_official_rulebook_v2_0_0(
    destination: Path,
    *,
    acknowledge_reference_only: bool,
    progress: Callable[[int, int], None] | None = None,
    timeout_seconds: float = 60.0,
    guide: SurfaceGuideSource | None = None,
) -> DownloadedRulebook:
    """Download and verify the exact official PDF used by the local surface guide.

    ``guide`` selects the season's rulebook; the default is RMUC 2026 V2.0.0.
    """

    if not acknowledge_reference_only:
        raise DownloadError(
            "explicit acknowledgement required: the official rulebook is reference-only "
            f"material; review {OFFICIAL_RULE_CENTRE}"
        )
    requested = Path(destination).expanduser()
    if requested.is_symlink():
        raise DownloadError(f"destination must not be a symlink: {requested}")
    target = requested.resolve()
    url, expected_size, expected_sha256 = _rulebook_identity(guide)
    if target.exists():
        verified = verify_official_rulebook_v2_0_0(target, guide)
        return DownloadedRulebook(
            verified.path,
            verified.size_bytes,
            verified.sha256,
            reused=True,
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.parent.is_symlink():
        raise DownloadError(f"destination parent must not be a symlink: {target.parent}")

    request = Request(
        url,
        headers={"User-Agent": "rmuc-mujoco/0.2"},
    )
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{target.name}.",
            suffix=".part",
            dir=target.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
                size, digest = _copy_and_hash(
                    response,
                    temporary,
                    expected_size=expected_size,
                    progress=progress,
                )
        if size != expected_size:
            raise DownloadError(
                f"downloaded rulebook size mismatch: expected={expected_size}, actual={size}"
            )
        if digest != expected_sha256:
            raise DownloadError(
                f"downloaded rulebook SHA-256 mismatch: expected={expected_sha256}, actual={digest}"
            )
        assert temporary_path is not None
        if target.exists():
            raise DownloadError(
                f"destination appeared during download; refusing overwrite: {target}"
            )
        os.replace(temporary_path, target)
        temporary_path = None
        verified = verify_official_rulebook_v2_0_0(target, guide)
        return DownloadedRulebook(
            verified.path,
            verified.size_bytes,
            verified.sha256,
            reused=False,
        )
    except DownloadError:
        raise
    except Exception as exc:
        raise DownloadError(f"official rulebook download failed: {exc}") from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


__all__ = [
    "DownloadedRulebook",
    "DownloadedStep",
    "DownloadError",
    "LAST_REVIEWED_RULEBOOK_PUBLICATION_DATE",
    "LAST_REVIEWED_RULEBOOK_SHA256",
    "LAST_REVIEWED_RULEBOOK_SIZE",
    "LAST_REVIEWED_RULEBOOK_URL",
    "LAST_REVIEWED_RULEBOOK_VERSION",
    "LAST_RULE_REVIEW_DATE",
    "OFFICIAL_RULE_CENTRE",
    "OFFICIAL_RULEBOOK_V2_0_0_SHA256",
    "OFFICIAL_RULEBOOK_V2_0_0_SIZE",
    "OFFICIAL_RULEBOOK_V2_0_0_URL",
    "OFFICIAL_SOURCE_PAGE",
    "OFFICIAL_STEP_PRODUCT",
    "OFFICIAL_STEP_SHA256",
    "OFFICIAL_STEP_SIZE",
    "OFFICIAL_STEP_URL",
    "download_official_rulebook_v2_0_0",
    "download_official_step",
    "verify_official_rulebook_v2_0_0",
    "verify_official_step",
]
