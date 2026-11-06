"""Main CLI entry point.

No args    → run full pipeline (fetch → build → simulate → compare)
--build    → build transition datasets only
--simulate → run weekly simulations only
--compare  → compare existing simulations to markets only
"""
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from transitions.paths import ARTIFACTS, RESET

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("goalpost")


def _sources_ok() -> bool:
    manifest = Path("artifacts/sources/download-manifest.json")
    if not manifest.exists():
        return False
    import json
    report = json.loads(manifest.read_text())
    return report.get("status") == "complete"


def _transitions_ok() -> bool:
    return (ARTIFACTS / "reset" / "data" / "transition_set.json").exists()


def _find_latest_weekly() -> Path | None:
    weekly_dir = ARTIFACTS / "weekly"
    if not weekly_dir.exists():
        return None
    candidates = [
        d for d in weekly_dir.iterdir()
        if d.is_dir() and (d / "report.json").exists()
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.name, reverse=True)
    import json
    for cand in candidates:
        try:
            report = json.loads((cand / "report.json").read_text())
            if report.get("status") in ("complete", "incomplete"):
                return cand
        except (ValueError, OSError):
            continue
    return None


def _run_pipeline():
    """Orchestrate fetch → build → simulate → compare."""
    import io
    import json

    from transitions.fetch_sources import fetch
    from transitions.build_artifacts import build
    from transitions.weekly import main as weekly_main
    from cli.compare import run_compare

    # --- Step 1: Sources ---
    if not _sources_ok():
        logger.info("Sources incomplete — fetching ...")
        fetch(output="artifacts/sources")
    else:
        logger.info("Sources verified.")

    # --- Step 2: Transitions ---
    transitions_path = ARTIFACTS
    if not _transitions_ok():
        logger.info("Transitions missing — building ...")
        if transitions_path.exists():
            logger.info("Removing stale transitions directory ...")
            shutil.rmtree(transitions_path)
        build(sources=Path("artifacts/sources"), output=transitions_path)
    else:
        logger.info("Transitions verified.")

    data_path = RESET / "data"

    # --- Step 3: Weekly simulations ---
    export_path = _find_latest_weekly()
    if export_path is not None:
        logger.info("Found existing weekly export: %s", export_path.name)
    else:
        logger.info("No weekly export found.")

    # Always run weekly to ensure freshness; weekly creates a new timestamped dir
    logger.info("Running weekly simulations ...")
    argv = ["--data", str(data_path), "--simulations", "50000"]
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        rc = weekly_main(argv)
    except SystemExit as exc:
        rc = exc.code if isinstance(exc.code, int) else 1
    finally:
        output = sys.stdout.getvalue()
        sys.stdout = old_stdout
    if rc != 0:
        raise RuntimeError(f"Weekly simulation failed (exit {rc})")
    parsed = json.loads(output.strip().splitlines()[-1])
    export_path = Path(parsed["output"])
    logger.info("Weekly export written to: %s", export_path)

    # --- Step 4: Compare ---
    logger.info("Running market comparison ...")
    summary = run_compare(export_path=export_path)
    return summary


def main():
    # If first arg is --build, --simulate, or --compare, delegate
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "--build":
            sys.argv = [sys.argv[0]] + sys.argv[2:]
            from cli.build import main as build_main
            build_main()
            return
        elif cmd == "--simulate":
            sys.argv = [sys.argv[0]] + sys.argv[2:]
            from cli.simulate import main as simulate_main
            simulate_main()
            return
        elif cmd == "--compare":
            sys.argv = [sys.argv[0]] + sys.argv[2:]
            from cli.compare import main as compare_main
            compare_main()
            return

    # Default: run full pipeline
    try:
        _run_pipeline()
    except Exception as exc:
        logger.exception("Pipeline failed")
        key = __import__("os").environ.get("ODDS_API")
        msg = str(exc)
        if key:
            msg = msg.replace(key, "[redacted]")
        print(f"Pipeline failed: {msg}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
