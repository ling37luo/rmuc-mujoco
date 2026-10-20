from concurrent.futures import ThreadPoolExecutor
import json

import pytest

from rmuc_mujoco import regression
from rmuc_mujoco.cli import main


def test_suite_reports_failure_and_continues_remaining_sections(
    field_asset_dir, tmp_path, monkeypatch
):
    def broken(*args):
        raise RuntimeError("controller failed")

    monkeypatch.setattr(regression, "check_public_interfaces", broken)
    monkeypatch.setattr(
        regression, "check_surface_routes", lambda *a: {"status": "PASS", "ordinary_patches": []}
    )
    monkeypatch.setattr(
        regression,
        "run_stair_checks",
        lambda *a, **k: {"status": "PASS", "catalog": {"patches": []}},
    )
    monkeypatch.setattr(
        regression,
        "check_example_episodes",
        lambda *a: {"status": "PASS", "summary": {"episodes": 2, "completed": 0, "failed": 2}},
    )
    output = tmp_path / "suite.json"
    result = regression.run_regression(field_asset_dir, "source.json", output=output)
    assert result["status"] == "FAIL"
    assert result["failed_sections"] == ["interfaces"]
    assert set(result["unavailable_sections"]) == {
        "walls",
        "perimeter",
        "corners",
        "tunnels",
        "high_points",
    }
    assert result["sections"]["interfaces"]["error"] == "RuntimeError: controller failed"
    assert result["robot_task_outcomes"]["failed"] == 2
    assert json.loads(output.read_text()) == result
    before = output.read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        regression.run_regression(field_asset_dir, "source.json", output=output)
    assert output.read_bytes() == before


def test_missing_sections_and_contact_differences_cannot_be_passes():
    result = regression.summarize_sections({"corners": {"status": "NOT_AVAILABLE"}})
    assert result["status"] == "PARTIAL"
    with pytest.raises(ValueError, match="no regression sections"):
        regression.summarize_sections({})
    rows = [
        dict(case_id="route", profile=p, steps=10, state_contact_sha256=d)
        for p, d in (("full", "a"), ("collision_only", "b"))
    ]
    assert regression._parity(rows)[0]["status"] == "FAIL"


def test_check_cli_exposes_robot_outcomes_and_failure_exit_code(tmp_path, monkeypatch, capsys):
    received = {}

    def run(pack, source, **kwargs):
        received.update(kwargs)
        kwargs["progress"]("interfaces", "PASS")
        return dict(
            status="FAIL",
            suite_id=regression.SUITE_ID,
            passed_sections=["interfaces"],
            failed_sections=["corners"],
            unavailable_sections=[],
            robot_task_outcomes={"failed": 2},
        )

    monkeypatch.setattr(regression, "run_regression", run)
    assert (
        main(
            [
                "check",
                "pack",
                "--source-manifest",
                "source.json",
                "--output",
                str(tmp_path / "r.json"),
                "--profiles",
                "collision_only",
                "--speeds",
                "0.5",
            ]
        )
        == 2
    )
    assert received["profiles"] == ["collision_only"]
    assert received["speeds"] == [0.5]
    assert '"robot_task_outcomes"' in capsys.readouterr().out


def _fake_sections(monkeypatch, failing_profile=None):
    """Deterministic sections that record the routes they were handed."""

    seen = {}

    def section(name, asset, source_manifest, profiles, speeds, earlier):
        inputs = {key: value.get("catalog") for key, value in earlier.items()}
        seen.setdefault(name, []).append((inputs, tuple(profiles)))
        if name == "example_robots":
            return regression.check_example_episodes(
                asset, profiles, *regression._example_inputs(earlier)
            )
        if name == "corners":
            trials = [
                dict(
                    route_id="corner",
                    speed_m_s=speed,
                    profile=profile,
                    steps=3,
                    state_contact_sha256="x",
                    status="PASS",
                )
                for profile in profiles
                for speed in speeds
            ]
            return regression.corner_result({"routes": [1, 2, 3, 4], **inputs}, profiles, trials)
        return {"status": "PASS", "catalog": f"{name} routes", "ordinary_patches": []}

    def rows(asset, profile, *inputs):
        if profile == failing_profile:
            raise RuntimeError(f"{profile} crashed")
        return [
            dict(
                case_id="route",
                profile=profile,
                steps=3,
                state_contact_sha256="x",
                status="PASS",
                task_outcome="COMPLETE",
            )
        ]

    monkeypatch.setattr(regression, "_run_section", section)
    monkeypatch.setattr(regression, "_example_rows", rows)
    return seen


def _without_timings(report):
    return {
        **report,
        "sections": {
            name: {key: value for key, value in section.items() if key != "elapsed_s"}
            for name, section in report["sections"].items()
        },
    }


def test_parallel_sections_report_what_a_sequential_run_reports(
    field_asset_dir, tmp_path, monkeypatch
):
    seen = _fake_sections(monkeypatch)
    sequential = regression.run_regression(field_asset_dir, "source.json", output=tmp_path / "a")
    sequential_inputs = dict(seen)
    seen.clear()

    parallel = regression.run_regression(
        field_asset_dir,
        "source.json",
        output=tmp_path / "b",
        jobs=4,
        executor=ThreadPoolExecutor(max_workers=4),
    )

    assert _without_timings(parallel) == _without_timings(sequential)
    assert list(parallel["sections"]) == list(regression.SECTION_ORDER)
    profiles = tuple(parallel["profiles"])
    assert sequential_inputs["corners"] == [({"perimeter": "perimeter routes"}, profiles)]
    # Corners and example robots ran once per profile and merged in profile order.
    for name in regression.PROFILE_SECTIONS:
        [(inputs, _profiles)] = sequential_inputs[name]
        assert sorted(run[1] for run in seen[name]) == sorted((profile,) for profile in profiles)
        assert all(run[0] == inputs for run in seen[name])
    trials = parallel["sections"]["example_robots"]["trials"]
    assert [row["profile"] for row in trials] == list(profiles)
    corners = parallel["sections"]["corners"]
    assert [row["profile"] for row in corners["trials"]] == [p for p in profiles for _ in range(3)]
    assert len(corners["profile_parity"]) == 3 * (len(profiles) - 1)
    assert json.loads((tmp_path / "b").read_text()) == parallel


def test_a_failing_example_profile_fails_only_the_example_section(
    field_asset_dir, tmp_path, monkeypatch
):
    _fake_sections(monkeypatch, failing_profile="collision_only")

    report = regression.run_regression(
        field_asset_dir,
        "source.json",
        output=tmp_path / "r",
        jobs=2,
        executor=ThreadPoolExecutor(max_workers=2),
    )

    assert report["failed_sections"] == ["example_robots"]
    assert report["sections"]["example_robots"]["error"] == "RuntimeError: collision_only crashed"


def test_every_section_without_inputs_starts_first_in_parallel(field_asset_dir, tmp_path):
    independent = set(regression.SECTION_ORDER) - set(regression.SECTION_INPUTS)
    assert set(regression.PARALLEL_START_ORDER) == independent
    assert len(regression.PARALLEL_START_ORDER) == len(independent)
    with pytest.raises(ValueError, match="jobs"):
        regression.run_regression(field_asset_dir, "source.json", output=tmp_path / "r", jobs=0)


def test_profile_parts_merge_as_one_run_over_every_profile():
    def part(profile, digest="x", routes=(1, 2, 3, 4)):
        trial = dict(
            route_id="corner",
            speed_m_s=0.5,
            profile=profile,
            steps=3,
            state_contact_sha256=digest,
            status="PASS",
        )
        return regression.corner_result({"routes": list(routes)}, (profile,), [trial])

    merged = regression._merge_profiles(
        "corners", ("full", "pro"), [part("full"), part("pro", "y")]
    )
    assert merged["status"] == "FAIL"
    assert merged["profile_parity"] == [
        dict(route_id="corner", speed_m_s=0.5, profile="pro", status="FAIL")
    ]
    error = dict(status="ERROR", error="RuntimeError: pro crashed")
    assert regression._merge_profiles("corners", ("full", "pro"), [part("full"), error]) == error
    with pytest.raises(ValueError, match="different corner catalogs"):
        regression._merge_profiles(
            "corners", ("full", "pro"), [part("full"), part("pro", routes=[1])]
        )


def test_default_jobs_follow_the_cpu_count_and_available_memory(tmp_path, monkeypatch):
    monkeypatch.setattr(regression.os, "cpu_count", lambda: 16)
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal: 16000000 kB\nMemAvailable: 6000000 kB\n")
    assert regression.default_jobs(meminfo) == 2
    assert regression.default_jobs(tmp_path / "missing") == 8
    meminfo.write_text("MemAvailable: 100 kB\n")
    assert regression.default_jobs(meminfo) == 1
