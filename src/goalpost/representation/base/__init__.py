# Purpose: Re-export the shared representation interfaces from one package namespace.
# Public API: GameExtractor, ExtractedGame, TransitionBuilder, ImputationBuilder, TransitionModel.
# Consumers: Package API consumers; repository implementations import defining submodules.

from .builder import TransitionBuilder as TransitionBuilder
from .extractor import ExtractedGame as ExtractedGame
from .extractor import GameExtractor as GameExtractor
from .imputer import ImputationBuilder as ImputationBuilder
from .models import TransitionModel as TransitionModel
