"""Generated data lives outside the installed source tree."""
import os
from pathlib import Path
ARTIFACTS = Path(os.environ.get("GOALPOST_TRANSITION_DATA", "artifacts/transitions")).expanduser().resolve()
EXTRACTED = ARTIFACTS / "extracted"
RESET = ARTIFACTS / "reset"
GRACE = ARTIFACTS / "grace"
