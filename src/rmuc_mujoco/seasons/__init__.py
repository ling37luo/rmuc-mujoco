"""Official RMUC seasons this project can build fields from.

A season pins its official sources (the STEP drawing and the rulebooks used by
the builder) and lists the features that exist only on that season's field.
The season is chosen once when a field is built.  Every later command derives
it from the STEP identity recorded in the pack, so a pack cannot be read as a
different season.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Mapping

if TYPE_CHECKING:
    from ..scenarios import ScenarioSpec

# Season-only field features.  Core code offers a feature's scenarios and
# checks only when the pack's season lists it, and imports the season's code
# for it lazily, so other seasons never load it.
FLY_RAMPS = "fly_ramps"
TUNNELS = "tunnels"
HIGH_POINTS = "high_points"


@dataclass(frozen=True)
class OfficialFile:
    """An official download pinned by exact size and SHA-256."""

    url: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class ReviewedRulebook:
    """The newest rulebook checked for field changes, not a build input."""

    version: str
    publication_date: str
    file: OfficialFile
    review_date: str
    # (key, finding) pairs from comparing the field chapter with the build input.
    field_chapter_findings: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class TunnelRoof:
    """One tunnel roof, identified by its part in the build's source GLB.

    The covered span is the roof's lowest underside face: the heightfield is
    lowered to the channel floor there and the roof becomes a collision mesh.
    """

    tunnel_id: str
    source_part_index: int
    # CAD-frame search box (x0, x1, y0, y1) that contains exactly this roof.
    search_box_xy_m: tuple[float, float, float, float]
    # Expected covered-span size along CAD x and y, from the official drawing.
    covered_span_m: tuple[float, float]


@dataclass(frozen=True)
class HighPoint:
    """One 制高点: an 800 mm platform reached by climbing three 200 mm steps.

    Rulebook 4.2.4 places the high point in the base area and joins it to the
    trapezoid highland with three steps; figure 4-17 dimensions them as 200 mm
    risers with 300 mm treads.  The platform and its steps are drawn in the
    source and the collision heightfield already represents them, so the record
    pins where to measure and what the build must reproduce; it never changes
    field geometry.
    """

    high_point_id: str
    # CAD-frame box (x0, x1, y0, y1) around the platform and both step flights.
    search_box_xy_m: tuple[float, float, float, float]
    # Levels above the base-area floor around the high point, not above z = 0.
    platform_top_m: float
    step_rise_m: float
    step_tread_m: float
    # The level the trapezoid highland sits at, where the three-step flight lands.
    highland_level_m: float
    highland_step_count: int
    floor_step_count: int
    # Unit XY direction, from the platform centre, of each flight in the drawing.
    # Each flight descends along its own direction, so a climb runs the other way.
    highland_flight_xy: tuple[float, float]
    floor_flight_xy: tuple[float, float]


@dataclass(frozen=True)
class FieldSource:
    """One official STEP release of a season's field drawing."""

    season_id: str
    version: str
    step: OfficialFile
    step_product: str
    source_page: str
    # Visual mesh groups a correct conversion of this STEP produces; the pack
    # exporter rejects a source build with any other count.
    visual_mesh_count: int
    # The unsimplified GLB that the builder writes for this STEP; source-bound
    # checks address its parts by index.
    source_glb_sha256: str
    tunnel_roofs: tuple[TunnelRoof, ...] = ()
    high_points: tuple[HighPoint, ...] = ()


@dataclass(frozen=True)
class SurfaceGuideSource:
    """The rulebook page rendered as the optional overhead surface guide."""

    version: str
    publication_date: str
    file: OfficialFile
    overhead_pdf_page: int


@dataclass(frozen=True)
class Season:
    id: str
    title: str
    # Oldest first; the last source is what ``setup`` downloads by default.
    sources: tuple[FieldSource, ...]
    rule_centre: str
    last_reviewed_rulebook: ReviewedRulebook
    surface_guide: SurfaceGuideSource | None
    core_battlefield_size_m: tuple[float, float]
    features: frozenset[str]
    # Returns the season's scenario table; None means the core table only.
    scenario_table: Callable[[], Mapping[str, ScenarioSpec]] | None = None

    @property
    def default_source(self) -> FieldSource:
        return self.sources[-1]

    def has(self, feature: str) -> bool:
        return feature in self.features

    def require(self, feature: str) -> None:
        if feature not in self.features:
            raise ValueError(f"{self.title} fields have no {feature.replace('_', ' ')}")


from .rmuc2026 import SEASON as RMUC2026  # noqa: E402
from .rmuc2027 import SEASON as RMUC2027  # noqa: E402

SEASONS: dict[str, Season] = {season.id: season for season in (RMUC2026, RMUC2027)}
DEFAULT_SEASON_ID = RMUC2026.id


def season_ids() -> tuple[str, ...]:
    return tuple(SEASONS)


def get_season(season_id: str) -> Season:
    try:
        return SEASONS[season_id]
    except KeyError:
        known = ", ".join(SEASONS)
        raise ValueError(f"unknown season {season_id!r}; known seasons: {known}") from None


def source_for_step(sha256: object, size_bytes: object = None) -> FieldSource | None:
    """Return the registered source with this STEP identity, if any."""

    for season in SEASONS.values():
        for source in season.sources:
            if source.step.sha256 == sha256 and (
                size_bytes is None or source.step.size_bytes == size_bytes
            ):
                return source
    return None


def season_of(source: FieldSource) -> Season:
    return SEASONS[source.season_id]


__all__ = [
    "DEFAULT_SEASON_ID",
    "FLY_RAMPS",
    "HIGH_POINTS",
    "FieldSource",
    "HighPoint",
    "OfficialFile",
    "RMUC2026",
    "RMUC2027",
    "ReviewedRulebook",
    "SEASONS",
    "Season",
    "SurfaceGuideSource",
    "TUNNELS",
    "TunnelRoof",
    "get_season",
    "season_ids",
    "season_of",
    "source_for_step",
]
