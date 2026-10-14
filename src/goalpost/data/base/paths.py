# Purpose: define conventional directories for extracted and imputed transition artifacts.
# Public API: ARTIFACTS, EXTRACTED and IMPUTED are absolute Path values computed at import.
# Consumers: clients opting into these conventions; no current repository module imports them.
# Configuration: GOALPOST_TRANSITION_DATA overrides the artifacts/transitions default.
# Relative paths resolve from the importing process's working directory, and ~ is expanded.
# Changing the environment after import does not refresh these constants. No directories
# are created and no artifact contents or existence are validated here.

import os
from pathlib import Path

ARTIFACTS: Path = (
    Path(os.environ.get("GOALPOST_TRANSITION_DATA", "artifacts/transitions")).expanduser().resolve()
)
EXTRACTED: Path = ARTIFACTS / "extracted"
IMPUTED: Path = ARTIFACTS / "imputed"
