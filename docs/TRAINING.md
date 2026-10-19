# Training and evaluation

Use one verified field pack for full-field evaluation and local training
scenarios. The repository supplies geometry, reset/stepping interfaces and
measurements; your project supplies the robot, policy, rewards and optimizer.

The commands below assume the [quick start](../README.md#quick-start) is complete,
the environment is activated, and the working directory is the repository root.
`run` is headless. Use `view` when you want a window.

## Performance and parallel layout

Measure your completed pack with the included rover; no policy or CAD source is needed:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 rmuc-field benchmark PACK \
  --env-counts 1 4 16 --workers 1 4 --steps 2000 --repeats 3 \
  --output runs/performance.json
```

Each trial starts fresh processes, one model per worker and independent data per
environment. Only divisible layouts are run: 1×1, 1×4, 4×1, 1×16 and 4×4
(workers × environments per worker). A screened flat spawn and fixed four-second
stop/forward/turn/reverse cycle are shared by every profile. The report compares
every-step state fingerprints and sampled named contacts, and records load time,
aggregate steps/s, end-to-end throughput and summed worker RSS. Timing includes
control and diagnostics; it excludes policy inference and learning. File caches
are retained, and RSS can double-count shared pages. These are workload measurements,
not a universal training-capacity claim.

For a visible 1280×720 frame workload, on a desktop with OpenGL:

```bash
rmuc-field benchmark-view PACK --frames 240 --repeats 3 \
  --screenshots runs/performance-views --output runs/performance-view.json
```

This shows fixed overview, robot and fly-ramp views with flat lighting and no
livery. It measures physics, scene update, rendering and buffer presentation,
with completed GPU work and no frame cap. It excludes native MuJoCo UI panels;
the result is frame-processing capacity, not the monitor's refresh rate. Compare
the saved views as well as frame rates before choosing a lighter visual profile.

Start with `full` for viewing and `collision_only` for headless batches. More data
objects in one worker save repeated model allocations but do not provide CPU
parallelism. Compare multiple workers on the target host, then reuse the measured
layout in the external trainer. Keep detailed reports and screenshots local.

## One-command field regression

Retain the CAD intermediates when building with
`--keep-source-build ./runs/field-source`. Then run:

```bash
rmuc-field check ./runs/field-contacts \
  --source-manifest ./runs/field-source/manifest.json \
  --output ./runs/field-check.json
```

The versioned suite generates its routes from the selected pack and source.
It checks public loading/reset/stepping and manual/policy callbacks, ordinary
slopes, both fly ramps, isolated stairs, perimeter strips, whole-rover corner
approaches, source walls and tunnel roofs. It uses the included probes and example robots;
no private controller, saved route file or checkpoint is needed. All available
profiles are compared. Add `--profiles collision_only --speeds 0.5` for a shorter run.

`field-contacts` above is a fenced pack with the optional source-wall layer.
Starting from the quick-start fenced pack and its retained source build:

```bash
rmuc-field lite-pack ./runs/field-fenced ./runs/field-lite \
  --interactive-lite-faces 250000
rmuc-field source-wall-pack ./runs/field-source ./runs/field-lite ./runs/field-contacts
rmuc-field tunnel-pack ./runs/field-source ./runs/field-contacts ./runs/field-tunnels
```

Apply the tunnel layer last. Run `check` on `./runs/field-tunnels` to include it.

The suite also accepts other packs. Missing fence/wall layers are reported as
`NOT_AVAILABLE`, making the overall result `PARTIAL`; they are not silently
skipped as passes. Exit code 0 means all included field/health checks passed;
code 2 means failure or incomplete coverage. The JSON is updated after each
section and includes identities, routes, solver settings, contacts and failures.
Example robot `task_outcome` and top-level `robot_task_outcomes` separately report
completed, failed and timed-out actions. A healthy simulation can include a
robot that tips on stairs or a jump. GUI input/rendering and performance scaling
are outside this headless suite.

## Deck and perimeter contact

Retain the source build with `setup --keep-source-build ./runs/field-source`,
then check a pack that includes the perimeter proxy:

```bash
rmuc-field perimeter-check ./local-rmuc2026-field \
  --source-manifest ./runs/field-source/manifest.json --output ./runs/perimeter.json
```

This selects source-supported strips on all four sides and runs a 120 mm wheel
through angled approach, sliding in both directions, and retreat at 0.3/0.5/1.0 m/s.
Height is unconstrained. The report records penetration, support height, contacts,
acceleration, warnings and phase completion, and compares states and contacts
across all available profiles. Use `--profiles collision_only` for a shorter check.

`PASS` covers the selected strips. Corner panel joins are checked geometrically;
source-obstructed corner ground remains unverified for driving. A wheel pass does
not certify a robot that keeps driving into a wall. Reports remain local, and
the command does not modify the field pack.

For whole-rover corner approaches, use `corner-check` with the same arguments.
It finds supported paths from accepted perimeter entries, contacts the next
blocking structure, and reverses along the path. Source-obstructed literal
corners remain obstacles; the report identifies the reachable endpoint and
source/runtime support differences. Passing does not make the corner empty.

## Source walls and local corners

For a pack with source-derived wall collision, run:

```bash
rmuc-field wall-check ./local-rmuc2026-field \
  --source-manifest ./runs/field-source/manifest.json --output ./runs/walls.json
```

This compares wall meshes and transferred heightfield samples with the source,
then tests both sides, free ends and fence junctions with repeated contact and
retreat at 0.3/0.5/1.0 m/s. It also checks a local wheel approach at each corner,
recording the first blocking surface and return. Obstructed corners can contact
terrain before reaching the fence; these are not full-robot access routes.
Existing reference-floor differences are reported separately from dynamic passes.
All available profiles use identical inputs and are compared step by step.

RMUC 2026 draws its two road-area walls as closed convex parts, and the layer
uses them as drawn. RMUC 2027 draws the same walls merged with the road beside
them, so each wall mesh is the convex hull of its part's vertices inside a box
the season records. Every part triangle in the box lies on the hull; the hull
adds only the underside, which the terrain hides, and a 0.011 m² strip where the
road meets the wall's inner foot, at or below the road surface. Under a 2027
wall the heightfield follows that underside instead of dropping to the floor,
and the few wall-end samples the builder's spike filter left low are counted in
the transfer, so a 2027 build needs no separate wall-tip repair.

On a 2027 pack `wall-check` runs both sides and the fence junction of each wall.
Each wall's free end stands on the road block: below the wall's sloped bottom
edge the end face is the road's, owned by the heightfield, so the end-cap probes
are listed under `skipped_routes` rather than run. The two fence corners the
bases fill leave no supported start for the local probe; they are marked
`SOURCE_OBSTRUCTED`, and the corner section approaches them with the whole rover.

## Tunnels

Tunnel roofs are recorded by the heightfield as solid blocks. `tunnel-pack`
lowers the heightfield under each roof's covered span to the channel floor and
adds the roof slab as collision, both from the retained source build. The roofs
are pinned per official STEP release (RMUC 2026: two per side; RMUC 2027: three
per side, 800 mm channel, 350 mm metal roof, 250 mm clearance). Each replaced
sample must match the source roof, so a changed drawing is rejected.

```bash
rmuc-field tunnel-pack ./runs/field-source ./local-rmuc2026-field ./runs/field-tunnels
rmuc-field tunnel-check ./runs/field-tunnels --output ./runs/tunnels.json
```

`tunnel-check` slides a rigid block 30 mm lower than each roof's clearance along
the channel in both directions; it must clear the roof without touching it. A
block 30 mm taller must be stopped by the roof, and a ball must rest on its top.
The report lists the simulated clearance per roof. The builder raises ground below
the main floor to it, so a tunnel over tilted ground can show slightly less than
the official 250 mm. The probes do not evaluate robot gaits.

Rulebook 4.4.2 lets a robot pass through a tunnel or over it, and table 5-27
tells the two apart by whether it was detected inside on the way from one end
to the other. The same report therefore publishes two routes per roof for
`tunnel_basic`, along the roof's travel axis and ending 0.6 m past the roof on
each side: `*_through` must pass under the roof and `*_over` above it. The routes
share their ends, so the session fails a run whose body crosses the roof span on
the wrong side (`passed_over_roof`, `passed_under_roof`):

```bash
rmuc-field run ./runs/field-tunnels --scenario tunnel_basic   --route-catalog ./runs/tunnels.json --directions uphill downhill --speeds 0.3 0.5
rmuc-field view ./runs/field-tunnels --scenario tunnel_basic   --route-catalog ./runs/tunnels.json --patch tunnel_c_red_through
```

The included four-wheel rover is lower than the roofs, and it drives every
through route in both directions on both seasons; `check` runs those in its
example-robot section. Crossing over takes clearing a 0.27–0.44 m bar with an open
underside, which none of the included robots can do, so their over runs fail as
`passed_under_roof`. The rover's round trips through the B tunnels can stall: the
far end sits on the foot of the road ramp, and the rover turns back only within
5 cm of the end, which its small wheel torque cannot climb to.

## Isolated stairs and ledges

When building a 1 cm pack, add `--keep-source-build ./runs/field-source` to
`setup` or `build`. Source contact checks need that retained CAD build; the
normal build removes it after exporting the runtime pack.

```bash
rmuc-field stairs-check ./local-rmuc2026-field \
  --source-manifest ./runs/field-source/manifest.json --output ./runs/stairs.json
rmuc-field run ./local-rmuc2026-field --scenario stairs_basic \
  --route-catalog ./runs/stairs.json --directions uphill downhill roundtrip \
  --speeds 0.3 0.5 1.0 --workers 2
rmuc-field view ./local-rmuc2026-field --scenario stairs_basic \
  --route-catalog ./runs/stairs.json --direction downhill --control policy
```

`stairs-check` selects isolated risers from reviewed source parts, checks both
treads and the seam, then compares a 120 mm wheel against the field and a local
source-plane reference at 0.3/0.5/1.0 m/s. It records blocked approaches as well
as traversals. A matched blocked approach can pass the contact comparison;
it does not mean the wheel climbed the step. Field geometry is unchanged.

`run` and `view` use the same catalog, robot composition and direction tracking.
Add `--robot` and `--controller` to use your own robot, or `--patch ID` in the
viewer / `--patches ID...` in batches to select a route. `SlopeSession` accepts
`route_catalog` for Python use. Reports separate numerical health from traversal;
small example wheels are not expected to climb every source step.

Keep the report local and use a new filename for another contact check. The
catalog is bound to its field pack. Coverage is limited to selected isolated
steps; compound stairs, adjacent walls and multilevel structures are not certified.

## High points (制高点, RMUC 2027)

A 2027 high point is an 800 mm platform in each base area, joined to the
trapezoid highland by the three 200 mm steps of rulebook figure 4-17 section A-A,
with a four-step flight down to the base-area floor as section B-B. Both flights
are drawn in the official source and the collision heightfield already carries
them; `high-point-check` measures the built field against that record, probes it,
and publishes the climbs as routes in the same report:

```bash
rmuc-field high-point-check ./local-rmuc2027-field \
  --source-manifest ./runs/field-source/manifest.json --output ./runs/high_point.json
rmuc-field run ./local-rmuc2027-field --scenario high_point_basic \
  --route-catalog ./runs/high_point.json --directions uphill --speeds 0.3 0.5
rmuc-field view ./local-rmuc2027-field --scenario high_point_basic \
  --route-catalog ./runs/high_point.json --patch high_point_red_highland
```

The check measures every level and both flights, compares the measured treads and
risers with the recorded dimensions and with the source surface, then drops a
rigid block on each level and drives a sled at the outermost riser of each
flight. The block must rest where the pack declares the surface and at that
level's own design height; the sled must reach the riser's face and be held
there, because a 200 mm face stops a body while a 200 mm ramp lets it through.
A high point is climbed rather than driven, so no wheel probe is used: the
question is whether the steps exist and are level, not whether a given robot can
mount them. The report states `geometry_modified: false`; the source build is
read for the audit only.

Two routes come out of each high point, one per flight: `*_highland` starts on
the highland and climbs to the platform, `*_floor` starts on the base-area floor.
As with the stair routes, the named low end is where an uphill run starts and
`uphill_unit_xy` is the direction it drives, so `--directions uphill` drives at
the steps and `--directions downhill` starts on the platform.

Without `--robot`, `run` and `view` drive an included step rover on these
routes: 500 mm wheels, its mass over the front axle and a stiff wheel-speed loop.
It climbs and descends all four 2027 flights at 0.3 and 0.5 m/s. With its mass
centred, the rear wheels' torque against a riser lifts the front wheels off the
step above and it stalls or tips backwards; the four-wheel rover of the other
scenarios has 160 mm wheels and tips on a 200 mm riser. The step rover is a test
vehicle, not a team's RM robot, and the check above does not depend on it; use
`--robot` and `--controller` to drive your own. Route ends sit 0.6 m from the
nearest riser, so a robot about a metre long spawns clear of the flight, and
`check` drives every high-point route up and down with the step rover as part of
its example-robot section.

Each level is compared after removing a locally fitted platform plane. The
recovered battlefield carries a shallow camber — about 2.6 % in y, with opposite
signs at the two ends — so raw absolute heights drift by tens of millimetres
across one high point.

The highland flight's treads measure the figure's 300 mm reference to a tenth of
a millimetre. The floor flight's do not: the drawn B-B treads measure about
305 mm, and the check accepts them because its tread tolerance is 15 mm around
the 300 mm reference. Both flight totals still match their reference exactly
(600 mm and 800 mm), which is what the extra 5 mm per tread is absorbed into.
The report prints the measured treads, so the difference is visible rather than
rounded away.

## Ordinary slopes

Inspect the available routes, then run the included four-wheel rover:

```bash
rmuc-field slope-catalog ./local-rmuc2026-field
rmuc-field run ./local-rmuc2026-field --scenario slope_basic \
  --directions uphill downhill roundtrip --speeds 0.3 0.5 \
  --repeats 2 --workers 2
```

The catalog screens ordinary slopes below 15 degrees and excludes the fly-ramp
corridors. A screened patch is a route candidate, not a dynamic pass. Available
routes depend on the pack. `--patches ID...` selects catalog routes.

| Direction | Start | Completion |
| --- | --- | --- |
| `uphill` | Low endpoint | Reach the high endpoint |
| `downhill` | High endpoint | Reach the low endpoint |
| `roundtrip` | Low endpoint | Complete both legs in one episode |

Watch the automatic example on a selected route:

```bash
rmuc-field view ./local-rmuc2026-field \
  --scenario slope_basic --control policy --direction roundtrip --profile full
```

Human viewing defaults to uphill only. Separate uphill and downhill runs do
not count as a roundtrip. Batch reports retain each leg, failure and timeout.

## Fly ramps

Run both ramps or select `fly_ramp_north` / `fly_ramp_south`:

```bash
rmuc-field run ./local-rmuc2026-field --scenario fly_ramp \
  --speeds 1.5 2.0 2.5 --repeats 2 --workers 2
rmuc-field view ./local-rmuc2026-field \
  --scenario fly_ramp_north --control policy --profile full --camera spawn
```

The included rover is a working interaction example, not a guaranteed jump
policy. Reports follow approach, takeoff, flight, recontact and landing hold.
For fly-ramp batches, `status=PASS` means the simulation completed without
numerical or controller errors; `task_status` and episode outcomes report jump
success. Inspect impact forces as well as route completion.

Use `--approach-distances`, `--lateral-offsets` and `--heading-offsets-deg` for
route variations within the robot's available clearance. These change starts,
not the underlying field geometry.

## External controllers

For slope or fly-ramp runs, supply both `--robot ./robot.xml` and
`--controller ./controller.py:make`. The factory returns the same callback
used by `view`; see the [template](../examples/controller_template.py).
It may also expose:

- `configure_route(route, direction, speed)` to receive the episode route;
- `reset(model, data)` to reset policy state;
- `set_seed(seed)` for private random generators;
- `press_name(key)` / `release_name(key)` for manual input.

The callback owns actuator mapping and policy-update timing. A factory with a
`field_asset` keyword receives the selected field or training region for queries.

Turning has an explicit command interface. Copy the
[turn controller template](../examples/turn_controller_template.py) into your
project and implement `reset(model, data, spawn, seed)`,
`step(model, data, command, step_index)` and optional `observe(model, data)`.
The template writes zero actions and is not a turning controller by itself.
After providing your `robot.xml` and `turn_controller.py`:

```bash
rmuc-field run ./local-rmuc2026-field \
  --robot ./robot.xml --controller ./turn_controller.py:make \
  --scenario turn_basic --phase spin --workers 2 --envs-per-worker 2 \
  --duration 8 --seed 42 --telemetry runs/turn-spin.json
```

Phases are `spin`, `arc` and `reversal`. The runner selects screened flat starts,
uses a zero-action first policy interval, then advances physics at 500 Hz and
the controller at 100 Hz. Turning metrics describe the supplied robot and
controller, not a universal pass for every robot.

## Parallel execution and reports

| Runner | Parallel layout |
| --- | --- |
| Turning | `spawn` processes; one `MjModel` and independent `MjData` objects per worker |
| Slopes and fly ramps | One active environment per worker; increase `--workers`, keep `--envs-per-worker=1` |
| External Isaac trainer | Separate consumer of exported field data |

Start with a small worker count and measure throughput and memory before
increasing it. Summaries are the default; use `--trajectory-dir` for detailed
slope/fly trajectories. Those runners create timestamped reports under `runs/`
when `--telemetry` is omitted; give turning runs an explicit telemetry path.
Reports record field/profile/robot identities, seeds, solver settings,
contacts, failures, throughput and worker memory. Numerical health and task
success must both be checked when comparing policies.

Python integrations can use `MuJoCoScenario`, `SlopeSession`, `FlyRampSession`
and their batch helpers for reset, stepping and diagnostics.

## Local fly-ramp regions

Use the 1 cm fenced pack from the [perimeter step](../README.md#quick-start)
to include both terrain and perimeter contact in the optional PhysX probe.
Export the two ramps separately:

```bash
rmuc-field training-region ./runs/field-fenced ./runs/fly-regions/north \
  --scenario fly_ramp_north --approach-distance 0.9
rmuc-field training-region ./runs/field-fenced ./runs/fly-regions/south \
  --scenario fly_ramp_south --approach-distance 0.9
rmuc-field run ./runs/fly-regions/north --scenario fly_ramp_north \
  --speeds 2.0 --workers 2 --footprint-radius 0.35
```

Use new output directories for new exports. The crop copies source grid samples
without resampling, keeps the source perimeter if present, and excludes visual
meshes. Its manifest records source and profile hashes, bounds and the route.
Set `--footprint-radius` for your robot; 0.35 m above is an example.

`TrainingRegion.open()` and `compose_training_region_with_robot()` provide the
same region to Python consumers. Cropped edges are artificial training limits:
reset when `region.contains_footprint(x, y, radius)` returns false. They are not
new physical walls. Use the complete field for final policy evaluation.

## Isaac handoff

`load_isaac_heightfield(pack, scenario=..., profile="collision_only")` supplies
height samples, coordinates, routes and hashes to an external Isaac consumer.
`run --backend isaac` emits a descriptor; it does not start training.

The [Isaac guide](../adapters/isaac/README.md) shows region export and the optional
PhysX probe. Contact probes have been exercised at 1, 4 and 16 environments;
large-scale articulated training throughput and cross-backend friction response
remain unmeasured. Keep each exported profile's identity with training results.
