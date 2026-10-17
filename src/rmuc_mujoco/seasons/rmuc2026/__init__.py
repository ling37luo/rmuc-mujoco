"""RMUC 2026: pinned official sources and 2026-only field features."""

from __future__ import annotations

from .. import (
    FLY_RAMPS,
    TUNNELS,
    FieldSource,
    OfficialFile,
    ReviewedRulebook,
    Season,
    SurfaceGuideSource,
    TunnelRoof,
)


def _scenario_table():
    from .scenarios import SCENARIOS

    return SCENARIOS


STEP_V2_0_0 = FieldSource(
    season_id="rmuc2026",
    version="2.0.0",
    step=OfficialFile(
        url=(
            "https://hz-rm-bbs-web-prod.oss-cn-hangzhou.aliyuncs.com/"
            "f637e13eadfc4602a76c7a05c54f71271782803452461/RMUC2026_V2.0.0.stp"
        ),
        size_bytes=1_254_821_405,
        sha256="8dfe9ebd761e44d91361b3e593bc05416329112217b58cb35800b3cde2ffae33",
    ),
    step_product="00_RMUC2026_FINALS_ASM",
    source_page="https://bbs.robomaster.com/article/814728?source=8",
    visual_mesh_count=38,
    source_glb_sha256="5bc8042c0fcb90e6f23f1f6624dbf8af1177b582a56254b4df092ee04ef23705",
    # Two tunnels per side; the field is centrally symmetric about (0, 1.624).
    tunnel_roofs=(
        TunnelRoof("a_red", 554, (-5.4, -4.0, -4.1, -3.1), (0.70, 0.35)),
        TunnelRoof("a_blue", 557, (4.0, 5.4, 6.34, 7.34), (0.70, 0.35)),
        TunnelRoof("b_red", 388, (0.5, 1.4, -5.0, -3.6), (0.35, 0.80)),
        TunnelRoof("b_blue", 389, (-1.4, -0.5, 6.84, 8.24), (0.35, 0.80)),
    ),
    stair_parts=(235, 236, 382, 383, 476, 554, 556, 557, 558),
    # The fly ramps reach past both ends of the 28 m deck.
    cad_outer_extents_xy_m=(29.752008274927, 16.003130912781),
)

SEASON = Season(
    id="rmuc2026",
    title="RMUC 2026",
    sources=(STEP_V2_0_0,),
    rule_centre="https://bbs.robomaster.com/wiki/20204847/809871?source=7",
    last_reviewed_rulebook=ReviewedRulebook(
        version="V2.2.0",
        publication_date="2026-08-07",
        file=OfficialFile(
            url=(
                "https://hz-rm-bbs-web-prod.oss-cn-hangzhou.aliyuncs.com/"
                "e354d2750e17485e8f67e5b5a9d1a2891786094242879/"
                "RoboMaster%202026%20%E6%9C%BA%E7%94%B2%E5%A4%A7%E5%B8%88%E8%B6%85%E7%BA%A7"
                "%E5%AF%B9%E6%8A%97%E8%B5%9B%E6%AF%94%E8%B5%9B%E8%A7%84%E5%88%99%E6%89%8B"
                "%E5%86%8CV2.2.0%EF%BC%8820260807%EF%BC%89.pdf"
            ),
            size_bytes=21_018_604,
            sha256="88ae3c7d0bbd7312c8095eff2603910437e292a0b53f46c9378f26144c09a6f5",
        ),
        review_date="2026-09-15",
        field_chapter_findings=(("static_field_chapter_vs_v2_0_0", "NO_CHANGE_DETECTED"),),
    ),
    surface_guide=SurfaceGuideSource(
        version="V2.0.0",
        publication_date="2026-06-26",
        file=OfficialFile(
            url=(
                "https://bbs-web-static.robomaster.com/"
                "5fbd46b110e542faaea519a6dd3065761782460789174/"
                "RoboMaster%202026%20%E6%9C%BA%E7%94%B2%E5%A4%A7%E5%B8%88%E8%B6%85%E7%BA%A7%"
                "E5%AF%B9%E6%8A%97%E8%B5%9B%E6%AF%94%E8%B5%9B%E8%A7%84%E5%88%99%E6%89%8B%"
                "E5%86%8CV2.0.0%EF%BC%8820260626%EF%BC%89.pdf"
            ),
            size_bytes=21_012_597,
            sha256="59d65aac5bac75fd6bbab157e224eb936b49cd2b9d60381fbe196fad635b473a",
        ),
        overhead_pdf_page=32,
    ),
    core_battlefield_size_m=(28.0, 15.0),
    features=frozenset({FLY_RAMPS, TUNNELS}),
    scenario_table=_scenario_table,
)
