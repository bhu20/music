"""Symbolic SATB lattice and interpretable constraints.

All pitches are MIDI integers. Perfect intervals are approximated in equal-tempered
semitones; correct diatonic spelling is out of scope for this first experiment.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product, combinations
from functools import lru_cache
from math import ceil
import re

VOICE_NAMES = ("Soprano", "Alto", "Tenor", "Bass")
VOICE_RANGES = ((60, 84), (55, 77), (48, 67), (40, 60))
VOICE_CENTRES = (73, 65, 57, 48)
NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
ROOT_PCS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

PRESETS = {
    "Pop cadence (C G Am F G C)": "C G Am F G C",
    "Minor progression (Am F Dm E Am)": "Am F Dm E Am",
    "Classical cadence (C F Dm G C)": "C F Dm G C",
    "Longer (C Am F G Em Am Dm G C)": "C Am F G Em Am Dm G C",
}


@dataclass(frozen=True)
class Chord:
    name: str
    pitch_classes: frozenset[int]
    root: int


@dataclass(frozen=True)
class Voicing:
    pitches: tuple[int, int, int, int]  # soprano, alto, tenor, bass


@dataclass(frozen=True)
class Frame:
    chord: Chord
    bar: int
    beat: int
    meter: int


@dataclass(frozen=True)
class Features:
    parallel_fifths: int
    parallel_octaves: int
    oversized_leaps: int
    melodic_distance: float
    soprano_leap: int
    repeated_voices: int


def note_name(midi: int) -> str:
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def parse_progression(source: str) -> list[Chord]:
    tokens = re.split(r"[\s,|]+", source.strip())
    tokens = [x for x in tokens if x]
    if not 2 <= len(tokens) <= 20:
        raise ValueError("Enter 2–20 chords (e.g. C G Am F G C).")
    output: list[Chord] = []
    for name in tokens:
        match = re.fullmatch(r"([A-Ga-g])([#b]?)(maj7|m7|dim|aug|m|7)?", name)
        if not match:
            raise ValueError(f"Unsupported chord '{name}'. Try C, F#, Bbm, Ddim, G7, Cmaj7 or Am7.")
        letter, accidental, quality = match.groups()
        base = ROOT_PCS[letter.upper()]
        base += 1 if accidental == "#" else (-1 if accidental == "b" else 0)
        root = base % 12
        quality = quality or "major"
        shapes = {
            "major": (0, 4, 7), "m": (0, 3, 7), "dim": (0, 3, 6),
            "aug": (0, 4, 8), "7": (0, 4, 7, 10),
            "maj7": (0, 4, 7, 11), "m7": (0, 3, 7, 10),
        }
        pcs = frozenset((root + interval) % 12 for interval in shapes[quality])
        output.append(Chord(name, pcs, root))
    return output


def make_frames(chords: list[Chord], meter: int = 4, beats_per_bar: int = 1) -> list[Frame]:
    if meter not in (2, 3, 4) or not 1 <= beats_per_bar <= meter:
        raise ValueError("Meter must be 2, 3, or 4 and events/bar must not exceed the meter.")
    # Subdivision is uniform; event locations need not fall on all integer beats if 3 / 2.
    return [Frame(chord, i + 1, beat, meter)
            for i, chord in enumerate(chords)
            for beat in range(beats_per_bar)]


def static_errors(v: Voicing, chord: Chord, *, soprano_alto_gap: int = 12,
                  alto_tenor_gap: int = 12, tenor_bass_gap: int = 19,
                  root_in_bass: bool = False) -> tuple[str, ...]:
    pitches = v.pitches
    errors = []
    for name, p, (lo, hi) in zip(VOICE_NAMES, pitches, VOICE_RANGES):
        if not lo <= p <= hi:
            errors.append(f"{name} outside range")
        if p % 12 not in chord.pitch_classes:
            errors.append(f"{name} non-chord note")
    if not all(pitches[i] > pitches[i + 1] for i in range(3)):
        errors.append("voice crossing/equality")
    for i, gap in enumerate((soprano_alto_gap, alto_tenor_gap, tenor_bass_gap)):
        if pitches[i] - pitches[i + 1] > gap:
            errors.append(f"excessive {VOICE_NAMES[i]}–{VOICE_NAMES[i+1]} spacing")
    if not chord.pitch_classes.issubset({p % 12 for p in pitches}):
        errors.append("incomplete chord")
    if root_in_bass and pitches[3] % 12 != chord.root:
        errors.append("bass not root")
    return tuple(errors)


@lru_cache(maxsize=128)
def candidates(chord: Chord, cap: int = 100, max_sa: int = 12,
               max_at: int = 12, max_tb: int = 19,
               root_bass: bool = False, seed: int = 5) -> tuple[Voicing, ...]:
    """Build fixed candidate lattice shared across A/B/C/D.

    Deterministically stratify candidate voicings rather than sampling only the
    tightest chords. This cap makes DP exact *within this pruned lattice* only.
    """
    pools = [tuple(p for p in range(lo, hi + 1) if p % 12 in chord.pitch_classes)
             for lo, hi in VOICE_RANGES]
    accepted = []
    for s, a, t, b in product(*pools):
        if not (s > a > t > b and s-a <= max_sa and a-t <= max_at and t-b <= max_tb):
            continue
        if root_bass and b % 12 != chord.root:
            continue
        if not chord.pitch_classes.issubset({s % 12, a % 12, t % 12, b % 12}):
            continue
        accepted.append(Voicing((s,a,t,b)))
    if not accepted:
        return tuple()
    # Representative lattice: favour the comfortable middle but retain open voicings.
    # Every system receives precisely the same candidate list for fair comparisons.
    accepted.sort(key=lambda v: (
        sum(abs(v.pitches[i]-VOICE_CENTRES[i]) for i in range(4))
        + 0.14 * (v.pitches[0]-v.pitches[3]),
        v.pitches))
    if len(accepted) <= cap:
        return tuple(accepted)
    front_n = max(1, cap // 2)
    result = accepted[:front_n]
    tail = accepted[front_n:]
    remaining = cap-front_n
    # Spread remaining coverage across the sorted spectrum and vary by fixed seed.
    for i in range(remaining):
        index = min(len(tail)-1, int(((i + ((seed % 7) / 7.0)) / max(remaining, 1)) * len(tail)))
        result.append(tail[index])
    return tuple(dict.fromkeys(result))


def transition_features(prev: Voicing, curr: Voicing, max_leap: int = 7) -> Features:
    x, y = prev.pitches, curr.pitches
    fifths = octaves = 0
    for a, b in combinations(range(4), 2):
        m1 = y[a] - x[a]
        m2 = y[b] - x[b]
        if not (m1 and m2 and (m1 > 0) == (m2 > 0)):
            continue
        previous_interval = abs(x[a] - x[b]) % 12
        current_interval = abs(y[a] - y[b]) % 12
        fifths += int(previous_interval == current_interval == 7)
        octaves += int(previous_interval == current_interval == 0)
    moves = [abs(a-b) for a,b in zip(x,y)]
    oversized = sum(int(m > max_leap) for m in moves)
    repeated = sum(int(a==b) for a,b in zip(x,y))
    return Features(fifths, octaves, oversized, float(sum(moves)), moves[0], repeated)
