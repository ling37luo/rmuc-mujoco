"""The rulebook overhead render that each season's surface guide is cut from."""

from __future__ import annotations

from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
import pytest

from rmuc_mujoco import _conversion
from rmuc_mujoco._conversion import (
    FieldBuildError,
    _crop_overhead_render,
    _extract_rulebook_surface_guide,
)
from rmuc_mujoco.seasons import RMUC2027


def _render(size: tuple[int, int], frame: tuple[int, int, int, int]) -> Image.Image:
    """A white page image with a dark field frame at ``frame`` (l, t, r, b)."""

    pixels = np.full((size[1], size[0], 3), 255, dtype=np.uint8)
    left, top, right, bottom = frame
    pixels[top:bottom, left:right] = 40
    return Image.fromarray(pixels)


def test_the_margin_crop_finds_the_dark_frame() -> None:
    # 952 x 510 px is the 28x15 m battlefield aspect.
    cropped, crop = _crop_overhead_render(_render((1000, 560), (20, 10, 972, 520)))

    assert crop == (20, 10, 972, 520) and cropped.size == (952, 510)


def test_a_pinned_crop_must_match_the_given_aspect_and_stay_in_the_image() -> None:
    image = _render((981, 559), (48, 25, 951, 524))

    cropped, crop = _crop_overhead_render(
        image, pinned_crop=(62, 26, 936, 524), expected_aspect=28.0 / 16.0
    )
    assert crop == (62, 26, 936, 524) and cropped.size == (874, 498)
    with pytest.raises(FieldBuildError, match="比例"):
        _crop_overhead_render(image, pinned_crop=(62, 26, 936, 524))
    with pytest.raises(FieldBuildError, match="越界"):
        _crop_overhead_render(image, pinned_crop=(62, 26, 990, 524), expected_aspect=1.75)


def test_the_2027_guide_is_cut_from_its_page_and_pinned_crop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands = []

    def fake_pdfimages(command, **_kwargs):
        commands.append(command)
        prefix = command[-1]
        _render((981, 559), (48, 25, 951, 524)).save(f"{prefix}-004.png")
        Image.new("RGB", (122, 14)).save(f"{prefix}-000.png")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(_conversion.shutil, "which", lambda name: "/usr/bin/pdfimages")
    monkeypatch.setattr(_conversion.subprocess, "run", fake_pdfimages)
    bounds = [[-25.18, -12.96], [2.82, 3.043]]

    record = _extract_rulebook_surface_guide(
        tmp_path / "rulebook.pdf",
        tmp_path,
        cad_outer_xy_bounds_m=bounds,
        guide=RMUC2027.surface_guide,
    )

    assert commands[0][1:5] == ["-f", "30", "-l", "30"]
    assert record["source_pdf_page"] == 30
    assert record["source_image_size_px"] == [981, 559]
    assert record["source_image_crop_ltrb_px"] == [62, 26, 936, 524]
    assert record["source_image_crop_rule"] == "pinned_fit_of_render_edges_to_source_heightfield"
    assert (record["width_px"], record["height_px"]) == (874, 498)
    assert record["world_mapping"]["world_bounds_xy_m"] == bounds
    assert "official V1.0.0 rulebook" in record["claim_boundary"]
    with Image.open(tmp_path / record["file"]) as written:
        assert written.size == (874, 498)
