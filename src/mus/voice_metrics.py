from __future__ import annotations

from collections.abc import Mapping, Sequence
from statistics import median
from typing import Any

def paired_family_labels(voice_count: int) -> list[str]:
    """Group adjacent voices into families (v0/v1, v2/v3, ...)."""
    return [f"family_{idx // 2}" for idx in range(voice_count)]

def infer_comfort_ranges(
    pitch_matrix: Sequence[Sequence[int | None]],
    padding: int = 7,
) -> dict[int, tuple[int, int]]:
    """Estimate a compact playable band for each voice from observed medians."""
    ranges: dict[int, tuple[int, int]] = {}
    if not pitch_matrix:
        return ranges

    voice_count = max(len(row) for row in pitch_matrix)
    for voice_idx in range(voice_count):
        values = [
            row[voice_idx]
            for row in pitch_matrix
            if voice_idx < len(row) and row[voice_idx] is not None
        ]
        if not values:
            continue
        center = int(median(values))
        ranges[voice_idx] = (center - padding, center + padding)
    return ranges

def _crossing_pairs(row: Sequence[int | None]) -> list[tuple[int, int]]:
    active = [(idx, value) for idx, value in enumerate(row) if value is not None]
    pairs: list[tuple[int, int]] = []
    for left in range(len(active)):
        i, pitch_i = active[left]
        for right in range(left + 1, len(active)):
            j, pitch_j = active[right]
            # Index order assumes higher voices appear first.
            if pitch_i < pitch_j:
                pairs.append((i, j))
    return pairs

def _max_displacement(row: Sequence[int | None]) -> int:
    active = [(idx, value) for idx, value in enumerate(row) if value is not None]
    if len(active) <= 1:
        return 0

    sorted_by_pitch = sorted(active, key=lambda pair: (-pair[1], pair[0]))
    index_to_sorted_pos = {
        voice_idx: sorted_pos for sorted_pos, (voice_idx, _pitch) in enumerate(sorted_by_pitch)
    }

    max_disp = 0
    for natural_pos, (voice_idx, _pitch) in enumerate(active):
        disp = abs(index_to_sorted_pos[voice_idx] - natural_pos)
        max_disp = max(max_disp, disp)
    return max_disp

def compute_tangling_metrics(
    pitch_matrix: Sequence[Sequence[int | None]],
    family_labels: Sequence[str] | None = None,
    comfort_ranges: Mapping[int, tuple[int, int]] | None = None,
) -> dict[str, Any]:
    if not pitch_matrix:
        return {
            "events": 0,
            "crossings_total": 0,
            "crossings_avg": 0.0,
            "k": 0,
            "k_avg": 0.0,
            "k1": 0,
            "k1_avg": 0.0,
            "k2": 0,
            "k2_avg": 0.0,
            "k3": 0,
            "k3_avg": 0.0,
        }

    voice_count = max(len(row) for row in pitch_matrix)
    if family_labels is None:
        family_labels = [str(idx) for idx in range(voice_count)]

    k_values: list[int] = []
    k1_values: list[int] = []
    k2_values: list[int] = []
    k3_values: list[int] = []
    crossing_counts: list[int] = []

    for row in pitch_matrix:
        pairs = _crossing_pairs(row)
        crossing_counts.append(len(pairs))
        k_values.append(_max_displacement(row))

        cross_family = 0
        within_family = 0
        for i, j in pairs:
            if i >= len(family_labels) or j >= len(family_labels):
                cross_family += 1
            elif family_labels[i] == family_labels[j]:
                within_family += 1
            else:
                cross_family += 1
        k1_values.append(cross_family)
        k2_values.append(within_family)

        max_range_deviation = 0
        if comfort_ranges:
            for voice_idx, pitch in enumerate(row):
                if pitch is None:
                    continue
                if voice_idx not in comfort_ranges:
                    continue
                low, high = comfort_ranges[voice_idx]
                if pitch < low:
                    max_range_deviation = max(max_range_deviation, low - pitch)
                elif pitch > high:
                    max_range_deviation = max(max_range_deviation, pitch - high)
        k3_values.append(max_range_deviation)

    events = len(pitch_matrix)
    return {
        "events": events,
        "crossings_total": sum(crossing_counts),
        "crossings_avg": round(sum(crossing_counts) / events, 4),
        "k": max(k_values),
        "k_avg": round(sum(k_values) / events, 4),
        "k1": max(k1_values),
        "k1_avg": round(sum(k1_values) / events, 4),
        "k2": max(k2_values),
        "k2_avg": round(sum(k2_values) / events, 4),
        "k3": max(k3_values),
        "k3_avg": round(sum(k3_values) / events, 4),
    }
