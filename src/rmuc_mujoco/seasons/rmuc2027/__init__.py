"""RMUC 2027: pinned official sources.

The 2027 field replaces the 2026 fly ramps with a third pair of tunnels that
robots drive through or over (rulebook 4.4.2: 800 mm channel, 350 mm metal
roof, 250 mm clearance), and adds a high point reached by three steps.
"""

from __future__ import annotations

from .. import TUNNELS, FieldSource, OfficialFile, ReviewedRulebook, Season, TunnelRoof

STEP_V1_0_0 = FieldSource(
    season_id="rmuc2027",
    version="1.0.0",
    step=OfficialFile(
        url=(
            "https://bbs-web-static.robomaster.com/"
            "080ae7230f6c469c9bec9d2c892797b61791994813953/RMUC2027_V1.0.0.stp"
        ),
        size_bytes=653_443_279,
        sha256="a5ad3ede3b7674fc98f2e73b71717216383cb2b4b5dcdbe900c2038918bea8e0",
    ),
    step_product="00_RMUC20270001_ASM",
    source_page="https://bbs.robomaster.com/article/1970956",
    # Measured from the first local V1.0.0 build (2026-10-16).
    visual_mesh_count=41,
    source_glb_sha256="5117497d92424e8bfdfc5154354de5861af06c23a6bc5b977977122405e33e92",
    # Tunnels a and b keep their 2026 places; c replaces the fly ramp.
    tunnel_roofs=(
        TunnelRoof("a_red", 229, (-5.4, -4.0, -4.1, -3.1), (0.80, 0.35)),
        TunnelRoof("a_blue", 226, (4.0, 5.4, 6.34, 7.34), (0.80, 0.35)),
        TunnelRoof("b_red", 116, (0.5, 1.4, -5.0, -3.6), (0.35, 0.80)),
        TunnelRoof("b_blue", 123, (-1.4, -0.5, 6.84, 8.24), (0.35, 0.80)),
        TunnelRoof("c_red", 178, (-0.5, 0.5, -6.0, -4.6), (0.35, 0.80)),
        TunnelRoof("c_blue", 150, (-0.5, 0.5, 7.84, 9.24), (0.35, 0.80)),
    ),
)

SEASON = Season(
    id="rmuc2027",
    title="RMUC 2027",
    sources=(STEP_V1_0_0,),
    rule_centre="https://bbs.robomaster.com/wiki/20204847/1969149",
    last_reviewed_rulebook=ReviewedRulebook(
        version="V1.0.0",
        publication_date="2026-10-15",
        file=OfficialFile(
            url=(
                "https://bbs-web-static.robomaster.com/"
                "451e5737ba5d4463909a3013456a4e551792055647969/"
                "RoboMaster%202027%20%E6%9C%BA%E7%94%B2%E5%A4%A7%E5%B8%88%E8%B6%85%E7%BA%A7"
                "%E5%AF%B9%E6%8A%97%E8%B5%9B%E6%AF%94%E8%B5%9B%E8%A7%84%E5%88%99%E6%89%8B"
                "%E5%86%8CV1.0.0%EF%BC%8820261015%EF%BC%89.pdf"
            ),
            size_bytes=22_455_861,
            sha256="9e2eed9295536c6fd547b62d634df1d5e70a2e1e2250e5dff3521d557e5d54be",
        ),
        review_date="2026-10-16",
        field_chapter_findings=(
            ("static_field_chapter_vs_rmuc2026_v2_2_0", "CHANGED_FLY_RAMPS_TO_TUNNELS"),
        ),
    ),
    surface_guide=None,
    core_battlefield_size_m=(28.0, 15.0),
    features=frozenset({TUNNELS}),
)
