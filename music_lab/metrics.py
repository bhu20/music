"""Metrics with explicit names and units. Higher diversity != higher musical quality."""
from __future__ import annotations
from collections import Counter
from math import log
import pandas as pd
from .core import note_name
from .search import Result, METHODS


def _pitch_entropy(notes) -> float:
    counts = Counter(notes)
    n = len(notes)
    if not n:
        return 0
    return -sum((v/n)*log(v/n, 2) for v in counts.values())


def summarize(result: Result) -> dict:
    t = result.steps
    n = max(1, len(t))
    s = [x.pitches[0] for x in result.path]
    return {
        'System': result.method,
        'Parallel fifths (count)': sum(x.features.parallel_fifths for x in t),
        'Parallel octaves (count)': sum(x.features.parallel_octaves for x in t),
        'Large leaps (count)': sum(x.features.oversized_leaps for x in t),
        'Total violations (count)': sum(x.features.parallel_fifths+x.features.parallel_octaves+x.features.oversized_leaps for x in t),
        'Mean motion (semitones/transition)': round(sum(x.features.melodic_distance for x in t)/n,2),
        'Soprano distinct notes (count)': len(set(s)),
        'Soprano pitch entropy (bits)': round(_pitch_entropy(s), 3),
        'Mean uncertainty (0–1)': round(sum(x.uncertainty for x in t)/n,3),
        'Mean penalty multiplier (0–1)': round(sum(x.multiplier for x in t)/n,3),
        'Runtime (seconds)': round(result.runtime_s, 3),
        'DP final score (a.u.)': round(result.total_cost, 2),
    }


def notes_dataframe(result: Result) -> pd.DataFrame:
    rows = []
    for frame, voicing in zip(result.frames,result.path):
        row = {'System':result.method, 'Bar':frame.bar, 'Event in bar':frame.beat+1,
               'Chord':frame.chord.name, 'Meter': f'{frame.meter}/4'}
        for name, midi in zip(('Soprano','Alto','Tenor','Bass'),voicing.pitches):
            row[f'{name} MIDI'] = midi
            row[f'{name} note'] = note_name(midi)
        rows.append(row)
    return pd.DataFrame(rows)


def trace_dataframe(result: Result) -> pd.DataFrame:
    return pd.DataFrame([{
        'System':result.method,'Bar':t.bar,'Event in bar':t.beat,
        'Previous chord':t.previous_chord,'Chord':t.chord,
        'Parallel fifths':t.features.parallel_fifths,'Parallel octaves':t.features.parallel_octaves,
        'Large leaps':t.features.oversized_leaps,'Motion (semitones)':t.features.melodic_distance,
        'Model uncertainty (0–1)':t.uncertainty,
        'Penalty multiplier (0–1)':t.multiplier,
        'Model NLL':t.neural_nll,
        'Transition cost (a.u.)':t.transition_cost
    } for t in result.steps])
