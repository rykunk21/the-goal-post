# Purpose: expose the NFL source adapter through the data.nfl package entry point.
# Public API: NFLDataSource; acquisition and parsing run only when its methods are called.
# Consumers: source clients importing from goalpost.data.nfl; internal modules may import
# the concrete implementation directly from data.nfl.source.

from .source import NFLDataSource as NFLDataSource
