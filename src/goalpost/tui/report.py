# Purpose: Collect explicit report inputs and invoke the shared research report job.
# Public API: PROMPTS, generate_report, run_report_screen.
# Consumers: The Generate Report action in tui.main and interface parity tests.

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import ratatui as r

from goalpost.cli.report import MarketTableJob
from goalpost.data.base.odds import QuoteSource
from goalpost.data.nfl.odds import OddsAPISource, SQLiteQuoteSource
from goalpost.simulation.nfl.comparison import ComparisonSettings

PROMPTS: list[str] = [
    "Simulation export directory",
    "Quote source (live / database / validate)",
    "Database path (database mode only)",
    "Output directory (optional)",
    "Book (default fanduel)",
    "Bankroll (optional; hypothetical dollars)",
]


# Purpose: Validate explicit TUI source choices and invoke the same report job as the CLI.
# Callers: run_report_screen and CLI/TUI parity tests.
def generate_report(values: dict[str, str]) -> str:
    # Reject missing or conflicting inputs before constructing adapters or output paths.
    export = values.get(PROMPTS[0], "").strip()
    if not export:
        raise ValueError("Simulation export directory is required")
    mode = values.get(PROMPTS[1], "").strip().lower()
    if mode not in ("live", "database", "validate"):
        raise ValueError("Choose live, database or validate explicitly")
    database = values.get(PROMPTS[2], "").strip()
    if mode == "database" and not database:
        raise ValueError("Database path is required in database mode")
    if mode != "database" and database:
        raise ValueError("Database path only applies in database mode")
    # Blank optional fields preserve the shared job defaults; no bankroll is inferred.
    book = values.get(PROMPTS[4], "").strip() or "fanduel"
    bankroll = values.get(PROMPTS[5], "").strip()
    settings = ComparisonSettings(book=book, bankroll=float(bankroll) if bankroll else None)
    # Source construction follows input validation; only the shared job fetches quotes.
    source = _quote_source(mode, database, book)
    output_path = _output_path(values.get(PROMPTS[3], ""))
    # Validation and priced reports use the same job so both interfaces retain parity.
    result = MarketTableJob(
        Path(export).expanduser(), output_path, source, settings, validate_only=mode == "validate"
    ).run()
    if mode == "validate":
        return f"Validated {result['games']} games. Output: {output_path}"
    return (
        f"Research/regulation-only: {result['fresh_rows']} fresh rows, "
        f"{result['evaluated_rows']} evaluated. EV/Kelly table: "
        f"{output_path / 'current-market-sizing.csv'}; full report: {output_path / 'report.json'}"
    )


# Purpose: Construct the explicitly selected quote adapter without requesting prices.
# Callers: generate_report after source mode and database path validation.
def _quote_source(mode: str, database: str, book: str) -> QuoteSource | None:
    if mode == "live":
        return OddsAPISource(book)
    if mode == "database":
        return SQLiteQuoteSource(Path(database).expanduser(), book)
    return None


# Purpose: Resolve an explicit output path or select a unique timestamped default.
# Callers: generate_report; the shared job owns directory creation and overwrite refusal.
def _output_path(value: str) -> Path:
    output = value.strip()
    if output:
        return Path(output).expanduser()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return Path("artifacts/market-sizing") / timestamp


# Purpose: Collect report inputs without prompting for or persisting API credentials.
# Callers: tui.main Generate Report action; cancellation does not create a job.
def run_report_screen(
    term: r.Terminal, prompt: Callable[[r.Terminal, str, list[str]], dict[str, str] | None]
) -> str:
    # No key prompt, saved key or implicit paid request. Selecting live is explicit.
    values = prompt(
        term, "Generate Report — research/regulation only; live makes one paid request", PROMPTS
    )
    if values is None:
        return "Canceled"
    return generate_report(values)
