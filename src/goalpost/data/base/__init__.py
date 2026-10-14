# Purpose: expose shared game records and the data-source interface at the package boundary.
# Public API: DataSource, Game, Play and Possession are re-exported from their defining modules.
# Consumers: data-source clients using the package entry point; internal implementations
# may import the defining modules directly. Importing this package does not fetch data.

from .models import Game as Game
from .models import Play as Play
from .models import Possession as Possession
from .source import DataSource as DataSource
