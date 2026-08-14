# GoalPost

NFL drive modeling system. Learn team representations, predict drive transition models, simulate outcomes.

## Architecture

Five-stage pipeline with abstract contracts at each stage:

1. **DataSource** — pull raw data (nflverse, Sportradar, etc.)
2. **DriveExtractor** — group plays into drives, compute state transitions
3. **TeamRepresentation** — learn latent team vectors (VAE, contrastive, etc.)
4. **TransitionModel** — predict state-to-state probabilities given team matchups
5. **Simulator** — Monte Carlo forward simulation to price markets

## Quick Start

```bash
pip install -e .
```

## Project Structure

```
src/goalpost/
├── abc/              # Abstract base classes
├── domain/           # Domain models (Game, Drive, Play, etc.)
├── data/             # Data sources + drive extractors
└── representation/   # Team representation encoders
```

## Status

Empty scaffold — ABCs defined, implementations stubbed. Next: implement `NFLVerseSource.parse()` and `NFLDriveExtractor.extract()` to validate data flow.
