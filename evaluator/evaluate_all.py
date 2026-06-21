from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SUBMISSIONS_DIR = REPO_ROOT / "submissions"
OUTPUTS_DIR = REPO_ROOT / "outputs"
LEADERBOARD_PATH = REPO_ROOT / "leaderboard.json"
WORKER_TIMEOUT_S = 120


def submission_paths() -> list[Path]:
    if not SUBMISSIONS_DIR.exists():
        return []
    return sorted(
        path for path in SUBMISSIONS_DIR.glob("*.py") if not path.name.startswith("_")
    )


def run_worker(path: Path) -> dict[str, Any]:
    output_path = OUTPUTS_DIR / "raw" / f"{path.stem}.json"
    command = [
        sys.executable,
        "-m",
        "evaluator.worker",
        str(path),
        "--output",
        str(output_path),
    ]

    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=WORKER_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return {
            "name": path.stem,
            "score": -10000.0,
            "status": "timeout",
            "error": f"Submission exceeded {WORKER_TIMEOUT_S} seconds",
            "scenarios": {},
        }

    if not output_path.exists():
        return {
            "name": path.stem,
            "score": -10000.0,
            "status": "error",
            "error": completed.stderr
            or completed.stdout
            or "Worker produced no output",
            "scenarios": {},
        }

    with output_path.open("r", encoding="utf-8") as input_file:
        result: dict[str, Any] = json.load(input_file)

    for scenario_name, replay in result.get("replays", {}).items():
        replay_path = OUTPUTS_DIR / "replays" / f"{path.stem}_{scenario_name}.json"
        replay_path.parent.mkdir(parents=True, exist_ok=True)
        with replay_path.open("w", encoding="utf-8") as output_file:
            json.dump(replay, output_file, indent=2)
            output_file.write("\n")

    result.pop("replays", None)
    if completed.returncode != 0 and result.get("status") == "ok":
        result["status"] = "error"
        result["error"] = completed.stderr or "Worker exited with a failure code"
    return result


def merge_leaderboard(results: list[dict[str, Any]]) -> dict[str, Any]:
    entries = []
    for result in results:
        name = str(result["name"])
        entries.append(
            {
                "name": name,
                "score": float(result["score"]),
                "status": result.get("status", "ok"),
                "scenarios": result.get("scenarios", {}),
                "error": result.get("error"),
            }
        )

    sorted_entries = sorted(
        entries,
        key=lambda entry: (-float(entry["score"]), str(entry["name"])),
    )
    return {
        "updated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "entries": sorted_entries,
    }


def write_leaderboard(data: dict[str, Any]) -> None:
    with LEADERBOARD_PATH.open("w", encoding="utf-8") as output_file:
        json.dump(data, output_file, indent=2)
        output_file.write("\n")


def resolve_submission_path(name: str) -> Path:
    stem = name[:-3] if name.endswith(".py") else name
    path = SUBMISSIONS_DIR / f"{stem}.py"
    if not path.exists():
        available = ", ".join(candidate.stem for candidate in submission_paths())
        raise SystemExit(
            f"No submission named '{stem}'. Available: {available or 'none'}"
        )
    return path


def visualize_replays(name: str, scenario_names: list[str]) -> list[Path]:
    # Imported lazily so the batch run does not pull in matplotlib.
    from evaluator.visualize import plot_replay, render_replay

    artifacts: list[Path] = []
    for scenario_name in scenario_names:
        replay_path = OUTPUTS_DIR / "replays" / f"{name}_{scenario_name}.json"
        if not replay_path.exists():
            print(
                f"No replay at {replay_path}; skipping {scenario_name} visualization",
                file=sys.stderr,
            )
            continue
        with replay_path.open("r", encoding="utf-8") as input_file:
            replay: dict[str, Any] = json.load(input_file)
        gif_path = OUTPUTS_DIR / f"{name}_{scenario_name}.gif"
        plot_path = OUTPUTS_DIR / f"{name}_{scenario_name}.png"
        render_replay(replay, gif_path)
        plot_replay(replay, plot_path)
        artifacts.append(gif_path)
        artifacts.append(plot_path)
    return artifacts


def evaluate_all() -> None:
    paths = submission_paths()
    results = [run_worker(path) for path in paths]
    leaderboard = merge_leaderboard(results)
    write_leaderboard(leaderboard)

    print(json.dumps(leaderboard, indent=2))

    failed = [result for result in results if result.get("status") != "ok"]
    if failed:
        names = ", ".join(str(result["name"]) for result in failed)
        print(f"Skipped failed submissions: {names}", file=sys.stderr)


def evaluate_one(name: str) -> None:
    path = resolve_submission_path(name)
    result = run_worker(path)
    stem = str(result["name"])

    if result.get("status") != "ok":
        traceback_text = result.get("traceback")
        if traceback_text:
            print(traceback_text, file=sys.stderr)
        raise SystemExit(f"{stem} failed: {result.get('error') or 'unknown error'}")

    scenarios = result.get("scenarios", {})
    artifacts = visualize_replays(stem, list(scenarios))

    print(f"{stem}: {float(result['score'])}")
    for scenario_name, scenario in scenarios.items():
        print(f"  {scenario_name}: {scenario['score']}")
    for artifact in artifacts:
        print(f"  wrote {artifact}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate MPC submissions and update the leaderboard"
    )
    parser.add_argument(
        "submission",
        nargs="?",
        help=(
            "Name of a single submission to evaluate and visualize "
            "(for example 'baseline'). Omit to evaluate every submission."
        ),
    )
    args = parser.parse_args()

    if args.submission is None:
        evaluate_all()
    else:
        evaluate_one(args.submission)


if __name__ == "__main__":
    main()
