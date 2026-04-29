from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from statistics import mean
from typing import Any

from .models import Constraints, ProblemInstance
from .voice_metrics import (
    compute_tangling_metrics,
    infer_comfort_ranges,
    paired_family_labels,
)

try:
    from music21 import corpus
except Exception:  # pragma: no cover - runtime dependency guard
    corpus = None


def _require_music21() -> None:
    if corpus is None:
        raise RuntimeError(
            "music21 is not installed. Install dependencies with: pip install -r requirements.txt"
        )


def iter_chorales(
    start: int,
    end: int,
    numbering: str = "riemenschneider",
) -> Iterator[tuple[int, Any]]:
    _require_music21()
    if end <= start:
        return

    # music21 Iterator signatures differ across versions; keyword iteration is stable.
    iterator = corpus.chorales.Iterator(numberingSystem=numbering)
    for index, chorale in enumerate(iterator, start=1):
        if index < start:
            continue
        if index >= end:
            break
        yield index, chorale


def extract_chord_progression(chorale: Any) -> tuple[list[str], list[float]]:
    chordified = chorale.chordify()
    roots: list[str] = []
    durations: list[float] = []

    for block in chordified.recurse().getElementsByClass("Chord"):
        root = block.root()
        if root is None:
            continue
        if root.octave is None:
            octave = 4
        else:
            octave = int(root.octave)
        roots.append(f"{root.name}{octave}")
        durations.append(float(block.quarterLength))
    return roots, durations


def extract_starting_pitches(chorale: Any) -> list[int]:
    starts: list[int] = []
    for part in chorale.parts:
        first_pitch: int | None = None
        for item in part.flatten().notes:
            if item.isNote:
                first_pitch = int(item.pitch.midi)
                break
            if item.isChord and item.pitches:
                first_pitch = int(max(item.pitches).midi)
                break
        if first_pitch is None:
            first_pitch = 60
        starts.append(first_pitch)

    # Preserve expected ordering from highest to lowest voice.
    for idx in range(1, len(starts)):
        if starts[idx] >= starts[idx - 1]:
            starts[idx] = starts[idx - 1] - 1
    return starts


def expand_starting_pitches(base_pitches: list[int], target_voices: int) -> list[int]:
    if target_voices <= 0:
        raise ValueError("target_voices must be positive")
    if not base_pitches:
        return [60 - idx * 3 for idx in range(target_voices)]
    if target_voices == len(base_pitches):
        return list(base_pitches)
    if target_voices == 1:
        return [base_pitches[0]]

    if len(base_pitches) == 1:
        pitch = base_pitches[0]
        return [pitch - idx * 2 for idx in range(target_voices)]

    jitters = [0, 1, -1, 2, -2, 1, -1]
    expanded: list[int] = []
    upper = len(base_pitches) - 1
    for idx in range(target_voices):
        position = (idx * upper) / (target_voices - 1)
        left = int(position)
        right = min(left + 1, upper)
        frac = position - left
        estimate = round(base_pitches[left] * (1.0 - frac) + base_pitches[right] * frac)
        estimate += jitters[idx % len(jitters)]
        expanded.append(estimate)

    for idx in range(1, len(expanded)):
        if expanded[idx] >= expanded[idx - 1]:
            expanded[idx] = expanded[idx - 1] - 1
    return expanded


def _part_timeline(part: Any) -> list[tuple[float, float, int | None]]:
    timeline: list[tuple[float, float, int | None]] = []
    for item in part.flatten().notesAndRests:
        start = float(item.offset)
        end = start + float(item.quarterLength)
        if end <= start:
            continue

        pitch: int | None = None
        if item.isNote:
            pitch = int(item.pitch.midi)
        elif item.isChord and item.pitches:
            pitch = int(max(item.pitches).midi)
        timeline.append((start, end, pitch))
    return timeline

def build_pitch_matrix(chorale: Any, max_events: int | None = None) -> list[list[int | None]]:
    timelines = [_part_timeline(part) for part in chorale.parts]
    offsets = sorted({event[0] for timeline in timelines for event in timeline})
    if max_events is not None and max_events > 0:
        offsets = offsets[:max_events]

    pointers = [0 for _ in timelines]
    matrix: list[list[int | None]] = []
    for offset in offsets:
        row: list[int | None] = []
        for voice_idx, timeline in enumerate(timelines):
            if not timeline:
                row.append(None)
                continue

            pointer = pointers[voice_idx]
            while pointer + 1 < len(timeline) and timeline[pointer + 1][0] <= offset:
                pointer += 1
            pointers[voice_idx] = pointer

            start, end, pitch = timeline[pointer]
            row.append(pitch if start <= offset < end else None)
        matrix.append(row)
    return matrix

def _classify_forbidden_perfect(
    interval_semitones: int,
    forbidden_parallels: set[str],
) -> str | None:
    simple = abs(interval_semitones) % 12
    if "P5" in forbidden_parallels and simple == 7:
        return "P5"
    if "P8" in forbidden_parallels and simple == 0:
        return "P8"
    return None

def analyze_counterpoint_motion(
    pitch_matrix: list[list[int | None]],
    forbidden_parallels: list[str] | None = None,
) -> dict[str, Any]:
    if not pitch_matrix:
        return {
            "forbidden_parallel_motion": {
                "total": 0,
                "by_type": {"P5": 0, "P8": 0},
                "no_forbidden_parallels": True,
            },
            "bass_contrary_motion": {
                "events": 0,
                "contrary": 0,
                "similar": 0,
                "oblique": 0,
                "contrary_ratio_active": 0.0,
                "contrary_ratio_all": 0.0,
                "bass_mostly_contrary": False,
            },
        }

    forbidden = set(forbidden_parallels or ["P5", "P8"])
    parallel_total = 0
    parallel_counts = {"P5": 0, "P8": 0}

    contrary = 0
    similar = 0
    oblique = 0
    bass_events = 0

    for prev_row, row in zip(pitch_matrix, pitch_matrix[1:]):
        active_voice_count = min(len(prev_row), len(row))
        if active_voice_count <= 1:
            continue

        for i in range(active_voice_count - 1):
            p0_i = prev_row[i]
            p1_i = row[i]
            if p0_i is None or p1_i is None:
                continue

            for j in range(i + 1, active_voice_count):
                p0_j = prev_row[j]
                p1_j = row[j]
                if p0_j is None or p1_j is None:
                    continue

                move_i = p1_i - p0_i
                move_j = p1_j - p0_j
                if move_i == 0 or move_j == 0:
                    continue
                if move_i * move_j <= 0:
                    continue

                start_class = _classify_forbidden_perfect(p0_i - p0_j, forbidden)
                end_class = _classify_forbidden_perfect(p1_i - p1_j, forbidden)
                if start_class is None or end_class is None:
                    continue
                if start_class != end_class:
                    continue

                parallel_total += 1
                parallel_counts[start_class] = parallel_counts.get(start_class, 0) + 1

        soprano_prev = prev_row[0]
        soprano_curr = row[0]
        bass_prev = prev_row[active_voice_count - 1]
        bass_curr = row[active_voice_count - 1]

        if (
            soprano_prev is None
            or soprano_curr is None
            or bass_prev is None
            or bass_curr is None
        ):
            continue

        bass_events += 1
        soprano_move = soprano_curr - soprano_prev
        bass_move = bass_curr - bass_prev
        if soprano_move == 0 or bass_move == 0:
            oblique += 1
        elif soprano_move * bass_move < 0:
            contrary += 1
        else:
            similar += 1

    active_motion_events = contrary + similar
    contrary_ratio_active = (
        0.0 if active_motion_events == 0 else contrary / active_motion_events
    )
    contrary_ratio_all = 0.0 if bass_events == 0 else contrary / bass_events

    return {
        "forbidden_parallel_motion": {
            "total": parallel_total,
            "by_type": {
                "P5": parallel_counts.get("P5", 0),
                "P8": parallel_counts.get("P8", 0),
            },
            "no_forbidden_parallels": parallel_total == 0,
        },
        "bass_contrary_motion": {
            "events": bass_events,
            "contrary": contrary,
            "similar": similar,
            "oblique": oblique,
            "contrary_ratio_active": round(contrary_ratio_active, 4),
            "contrary_ratio_all": round(contrary_ratio_all, 4),
            "bass_mostly_contrary": contrary_ratio_active >= 0.5 and active_motion_events > 0,
        },
    }

def extract_voice_leading_patterns(chorale: Any) -> dict[str, Any]:
    intervals: list[int] = []
    leaps = 0
    resolved_leaps = 0

    for part in chorale.parts:
        pitches: list[int] = []
        for item in part.flatten().notes:
            if item.isNote:
                pitches.append(int(item.pitch.midi))
            elif item.isChord and item.pitches:
                pitches.append(int(max(item.pitches).midi))

        part_intervals = [
            pitches[idx] - pitches[idx - 1] for idx in range(1, len(pitches))
        ]
        intervals.extend(part_intervals)

        for idx in range(1, len(part_intervals)):
            prev_motion = part_intervals[idx - 1]
            motion = part_intervals[idx]
            if abs(prev_motion) >= 4:
                leaps += 1
                if prev_motion * motion < 0 and abs(motion) <= 2:
                    resolved_leaps += 1

    if not intervals:
        return {
            "interval_count": 0,
            "mean_abs_interval": 0.0,
            "stepwise_ratio": 0.0,
            "leap_resolution_ratio": 0.0,
            "interval_histogram": {},
        }

    histogram: Counter[str] = Counter()
    for value in intervals:
        if value < -12:
            key = "lt_-12"
        elif value > 12:
            key = "gt_12"
        else:
            key = str(value)
        histogram[key] += 1

    abs_intervals = [abs(value) for value in intervals]
    step_count = sum(1 for value in abs_intervals if value <= 2)
    leap_ratio = 0.0 if leaps == 0 else resolved_leaps / leaps

    return {
        "interval_count": len(intervals),
        "mean_abs_interval": round(mean(abs_intervals), 4),
        "stepwise_ratio": round(step_count / len(intervals), 4),
        "leap_resolution_ratio": round(leap_ratio, 4),
        "interval_histogram": dict(sorted(histogram.items())),
    }

def try_extract_roman_numerals(riemenschneider_number: int) -> list[str] | None:
    _require_music21()
    candidates = [
        f"bach/choraleAnalyses/riemenschneider{riemenschneider_number:03d}.rntxt",
        f"bach/choraleAnalyses/riemenschneider{riemenschneider_number}.rntxt",
        f"bach/choraleAnalyses/{riemenschneider_number:03d}.rntxt",
    ]

    for entry in candidates:
        try:
            analysis = corpus.parse(entry)
        except Exception:
            continue
        figures = [
            str(item.figure)
            for item in analysis.recurse().getElementsByClass("RomanNumeral")
        ]
        if figures:
            return figures
    return None

def make_chorale_problem(
    chorale: Any,
    problem_id: str,
    target_voices: int,
    base_constraints: Constraints | None = None,
    include_roman: bool = False,
    riemenschneider_number: int | None = None,
    max_events_for_metrics: int | None = 256,
) -> ProblemInstance:
    roots, durations = extract_chord_progression(chorale)
    starting_pitches_4 = extract_starting_pitches(chorale)
    expanded_starts = expand_starting_pitches(starting_pitches_4, target_voices)

    pitch_matrix = build_pitch_matrix(chorale, max_events=max_events_for_metrics)
    family_labels = paired_family_labels(len(starting_pitches_4))
    comfort_ranges = infer_comfort_ranges(pitch_matrix, padding=7)
    tangling = compute_tangling_metrics(
        pitch_matrix,
        family_labels=family_labels,
        comfort_ranges=comfort_ranges,
    )
    counterpoint_checks = analyze_counterpoint_motion(
        pitch_matrix,
        forbidden_parallels=(base_constraints or Constraints()).forbidden_parallels,
    )

    constraints = base_constraints or Constraints()
    constraints = replace(
        constraints,
        max_k=constraints.max_k if constraints.max_k is not None else tangling["k"],
        max_k1=constraints.max_k1 if constraints.max_k1 is not None else tangling["k1"],
        max_k2=constraints.max_k2 if constraints.max_k2 is not None else tangling["k2"],
        max_k3=constraints.max_k3 if constraints.max_k3 is not None else tangling["k3"],
    )

    metadata: dict[str, Any] = {
        "source": "music21_corpus",
        "original_voice_count": len(starting_pitches_4),
        "observed_tangling": tangling,
        "voice_leading_patterns": extract_voice_leading_patterns(chorale),
        "counterpoint_checks": counterpoint_checks,
    }

    if riemenschneider_number is not None:
        metadata["riemenschneider_number"] = riemenschneider_number
    if include_roman and riemenschneider_number is not None and riemenschneider_number <= 20:
        metadata["roman_numerals"] = try_extract_roman_numerals(riemenschneider_number)

    return ProblemInstance(
        problem_id=problem_id,
        voices=target_voices,
        chord_roots=roots,
        chord_durations=durations,
        starting_pitches=expanded_starts,
        constraints=constraints,
        metadata=metadata,
    )

def write_instances(
    instances: list[ProblemInstance],
    out_dir: Path,
) -> tuple[Path, Path]:
    from .models import write_json, write_jsonl

    out_dir.mkdir(parents=True, exist_ok=True)
    instances_dir = out_dir / "instances"
    instances_dir.mkdir(parents=True, exist_ok=True)

    jsonl_rows: list[dict[str, Any]] = []
    for instance in instances:
        payload = instance.to_dict()
        write_json(instances_dir / f"{instance.problem_id}.json", payload)
        jsonl_rows.append(payload)

    jsonl_path = out_dir / "chorale_instances.jsonl"
    write_jsonl(jsonl_path, jsonl_rows)
    summary_path = out_dir / "summary.json"
    write_json(
        summary_path,
        {
            "count": len(jsonl_rows),
            "output_jsonl": str(jsonl_path),
            "output_instances_dir": str(instances_dir),
        },
    )
    return jsonl_path, summary_path
