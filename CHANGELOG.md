# Changelog

User-visible changes by release. Generated field packs and official assets are
built locally and are not included in releases. Passing individual scenarios
does not remove the full field's `DRAFT_BLOCKED` status.

## [2.0.0] - 2026-10-18

- Renamed the package to `rmuc-mujoco` (`rmuc_mujoco`, `rmuc-field`) for
  multi-season use. `rmuc2026_mujoco`, `rmuc2026-field` and `rmuc2026-mujoco`
  remain as aliases; pack formats and data identifiers are unchanged.
- Added a season registry (`rmuc_mujoco.seasons`) and `--season` for `source`,
  `download`, `build` and `setup`. A pack's season follows from its STEP identity.
- New packs record `season` and `official_step_version`; 1.x packs are
  identified by their STEP hash. Field files exported from the same source
  build are unchanged.
- Moved the fly-ramp modules to `rmuc_mujoco.seasons.rmuc2026`; the 1.x import
  paths still resolve. Fly-ramp scenarios are offered only on RMUC 2026 packs.
- Registered RMUC 2027 (STEP V1.0.0, rulebook V1.0.0). Each STEP release pins its
  visual mesh count, build GLB and tunnel roofs.
- Added `tunnel-pack` and `tunnel-check`: tunnels in both seasons can be driven
  through and over. `check` reports a tunnel section; spawn screening avoids roofs.
- Added `high-point-check` and the `high_point_basic` scenario for the RMUC 2027
  high points. It measures every level and both flights of each high point,
  audits them against the source surface, drops a rigid block on every level and
  drives a sled at the outermost riser of each flight, then publishes the four
  climbs as route records that `run` and `view` accept. `check` reports a
  high-point section. Field geometry is unchanged.
  Measured against the drawing: the highland flight's 300 mm treads and both
  flights' 200 mm risers and 600/800 mm totals are reproduced, while the drawn
  floor flight's treads measure about 305 mm. The check compares each tread
  against the 300 mm reference with a 15 mm tolerance, so the report prints the
  measured value instead of rounding the difference away.
- `run` accepts any published route catalog by the `scenario_id` it names, so a
  `high-point-check` report drives `high_point_basic` batches and viewer sessions
  directly. The trajectory-directory guard names the scenario it found.
- Without `--robot`, `high_point_basic` drives a big-wheel step rover that climbs
  and descends the 200 mm risers; other scenarios keep the four-wheel rover.
  High-point route ends sit 0.6 m from the nearest riser instead of 0.35 m, so a
  robot about a metre long spawns clear of the flight, and `check` drives the
  high-point routes in its example-robot section.
- `scenarios --asset` lists the bound pack's own season: a 2027 pack lists
  `high_point_basic` and no fly ramps instead of failing on a 2026-only entry.
- Source-bound checks accept a retained source build only for the field pack's
  own STEP release, through one shared gate. Each release pins its reviewed
  stair parts; `check` reports stairs as unavailable for a release without them
  instead of failing on another release's part numbers.
- RMUC 2027 records its two reviewed stair parts, so `stairs-check` and `check`
  cover the two central 200 mm steps on 2027 packs as they do on 2026 packs.
- `fence-pack` places the perimeter fence on RMUC 2027 packs. Each STEP release
  records the CAD outline the fence is calibrated to: the 2026 outline reaches
  past the deck to the fly ramps, 2027's is the 28 m deck itself. On a fenced
  2027 pack `check` runs the perimeter and corner sections.
- `source-wall-pack` and `wall-check` work on RMUC 2027 packs. Each STEP release
  records its source walls. RMUC 2027 merges the road-area walls with the road
  beside them, so each wall is the convex hull of its part's vertices in a
  recorded box, refused if a part triangle there leaves the hull or the hull
  adds side area the record does not account for. The RMUC 2026 wall pack is
  unchanged byte for byte.
- Added the `tunnel_basic` scenario for seasons with tunnels. `tunnel-check`
  publishes a through and an over route along each roof's travel axis; the
  session fails a run whose body crosses the roof span on the wrong side, and
  `check` drives the through routes with the four-wheel rover.

## [1.3.0] - 2026-10-09

- Added `benchmark` with fresh parallel workers, a fixed public rover workload,
  per-profile state/contact comparison, load time, throughput and memory reports.
- Added `benchmark-view` for visible GLFW frame measurements and comparable
  screenshots; results distinguish frame capacity from native viewer UI refresh.
- Added selectable Isaac contact-probe matrix counts through 64 environments.

- Added whole-rover corner approach, obstacle contact and reverse-route checks,
  with footprint-screened paths from accepted perimeter entries.
- Added `check`, a portable static-field regression using included robots,
  generated routes, public reset/control interfaces and profile comparisons.
  Reports keep unavailable features and robot task failures explicit.
- Fixed spawn screening inside mesh-owned walls and outside the fence interior.
- Added source wall ownership, repeated side/end contact and local corner checks,
  including state/contact comparisons across field profiles.
- Added source-screened perimeter approach, bidirectional sliding and retreat
  checks, with contact/state comparisons across field profiles.
- Added source-bound isolated stair checks, paired wheel/source contact results,
  and `stairs_basic` viewing/batches through the existing robot controller interface.
- Added optional `--keep-source-build` for reproducible local CAD contact audits.

- Shortened setup and integration documentation; separated training examples
  into one guide and updated the remaining full-field tasks.
- Removed machine-specific experiment reports from the public documentation.

## [1.2.1] - 2026-09-26

- Added optional source-derived wall collision for parts 402/403 and an
  `interactive_lite` visual profile sharing the pack's physics.
- Added wheel-route checks and multi-environment profile comparisons.
- Fixed robot key forwarding, interactive stepping and duplicate L/G toggles.
- Wall coverage remains partial. Lite is experimental and has not met the
  resource-improvement threshold for a recommended profile.

## [1.2.0] - 2026-09-23

- Added north/south fly-ramp episodes, an example rover and parallel evaluation
  across speeds and starting poses. Reports separate numerical health from
  takeoff, flight and landing success.
- Added source-bound 1 cm fly-ramp regions with routes, hashes and cropped
  perimeter collision for independent MuJoCo environments.
- Added Isaac region export and a separate Isaac Sim 6.1 PhysX contact probe
  for 1, 4 and 16 environments. Robot landing and friction equivalence remain
  outside that probe's scope.

## [1.1.0] - 2026-09-23

- Added screened `turn_basic` starts, spin/arc/reversal phases and parallel
  MuJoCo execution with external controllers.
- Added ordinary-slope route screening, an example rover and interactive or
  automatic route evaluation.
- Added separate uphill, downhill and roundtrip outcomes, contact diagnostics
  and optional trajectories. A roundtrip requires both legs in one episode.
- Added headless slope matrices across routes, directions, speeds and repeats.

## [1.0.0] - 2026-09-22

- Added the public scenario registry, external controller callbacks and shared
  robot composition for viewer and headless runs.
- Added telemetry for state, contacts, warnings, asset identities and resources.
- Added the optional Isaac heightfield descriptor and fixed fly-ramp wheel probe.
- Added robot, controller, scenario, headless and telemetry CLI options.

## [0.3.1] - 2026-09-21

- Added the missing `rtree` test dependency for clean CI environments.

## [0.3.0] - 2026-09-21

- Aligned the perimeter proxy with the inferred 28 × 15 m raised deck, with
  25 mm inward overlap to close the deck-to-fence trap.
- Added the v5 perimeter contract while retaining v1–v4 compatibility and
  fence `solref="0.04 1"`.

## [0.3.0a4] - 2026-09-21

- Aligned perimeter outer faces with the heightfield bounds to remove the
  exterior traversable strip. Added the v4 perimeter contract.

## [0.3.0a3] - 2026-09-20

- Adjusted north/south perimeter placement to improve fly-ramp clearance.
- Softened fence contact to `solref="0.04 1"` without changing terrain contact.
  Added the v3 perimeter contract.

## [0.3.0a2] - 2026-09-20

- Added `fence-pack` for optional four-sided physical containment and manifest
  verification of its collision geometry.
- Added a source-bound ramp audit distinguishing adjacent CAD structures from
  heightfield sampling errors.

## [0.3.0a1] - 2026-09-20

- Added experimental schema 3 outer-edge masks and a read-only boundary guard.
  The finite void surrogate does not provide true holes or safe edge traversal.
- Added source-bound wall/edge audit tools and a seven-sample wall-end repair
  for the audited 1 cm source grid.
- Set the default visual budget to 2.3 million faces and added an optional
  primitive energy-unit probe. Experimental contact candidates stay opt-in.

## [0.2.1] - 2026-09-20

- Improved livery conformity with adaptive visual refinement and reduced its
  lift to 5 mm. Collision geometry and materials are unchanged.

## [0.2.0] - 2026-09-19

- Recorded collision sample provenance and structural audit metadata.
- Improved livery sampling and made rulebook page margins transparent.
- Verified that the bootstrap PNG matches the floating-point heightfield.
- Distinguished size-only checks from hash verification and fixed the X11 test
  dependency in clean environments.

## [0.2.0a1] - 2026-09-15

- Added schema 2, named field lighting, L/G display controls and optional
  collision-free rulebook livery.
- Added 1 cm heightfields, fixed-ramp probes, optional bounded ramp refinement,
  local surface queries and deterministic spawn screening.
- Added visual mesh cleanup, even default lighting and overview/spawn cameras.
- Fixed friction-preset precedence and viewer shutdown ordering.
- Strengthened MJCF, heightfield and archive verification while retaining
  schema 1 compatibility. Fine sampling preserves the field coordinate frame.

## [0.1.0] - 2026-09-15

- Added local field construction from pinned official sources, manifest/hash
  verification, exact heightfield loading and robot-only MJCF composition.
- Added queries, CLI inspection, a viewer and synthetic-only CI tests.
- The single-heightfield collision model does not represent underpasses,
  stacked surfaces or dynamic mechanisms.
