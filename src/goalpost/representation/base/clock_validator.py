# Purpose: Offer segment coalescing and clock-order checks for mapping-based event data.
# Public API: coalesce_zero_time and validate_clock_progress.
# Consumers: Available representation utilities; no direct in-repository callers yet.

from typing import Any


# Purpose: Fold zero-elapsed segment yards into the preceding retained segment.
# Callers: Representation consumers explicitly choosing segment coalescing.
# Retained mappings are shared with the input and may be mutated; other fields are dropped
# from folded segments. A leading zero-elapsed segment is retained. Missing elapsed is zero.
def coalesce_zero_time(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged = []
    for seg in segments:
        if seg.get("elapsed", 0) == 0 and merged:
            merged[-1]["yards"] = merged[-1].get("yards", 0) + seg.get("yards", 0)
        else:
            merged.append(seg)
    return merged


# Purpose: Reject increasing remaining-clock values, or clamp them in place when non-strict.
# Callers: Representation consumers explicitly choosing this adjacent-event check.
# Missing remaining values compare as zero; this utility does not handle period resets.
# Clamping requires the preceding remaining key and raises KeyError if that key is absent.
def validate_clock_progress(events: list[dict[str, Any]], strict: bool = True) -> None:
    for i in range(1, len(events)):
        prev = events[i - 1]
        curr = events[i]
        if curr.get("remaining", 0) > prev.get("remaining", 0):
            if strict:
                raise ValueError("clock regression")
            else:
                curr["remaining"] = prev["remaining"]
