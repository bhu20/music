"""Comparable A/B/C/D Viterbi-style dynamic programming over a fixed voicing lattice."""
from __future__ import annotations
from dataclasses import dataclass
from math import log, cos, pi
from time import perf_counter
from .core import (Frame, Chord, Voicing, Features, candidates, make_frames,
                   transition_features, VOICE_CENTRES)

METHODS = {
    'A': 'A | Strict hard rules',
    'B': 'B | Fixed soft penalties',
    'C': 'C | Phrase-adaptive penalties',
    'D': 'D | Neural-uncertainty penalties',
}

@dataclass(frozen=True)
class Config:
    cap: int = 65
    meter: int = 4
    events_per_bar: int = 2
    max_leap: int = 7
    max_sa: int = 12
    max_at: int = 12
    max_tb: int = 19
    root_bass: bool = False
    seed: int = 5
    pf_weight: float = 3.8
    po_weight: float = 5.0
    leap_weight: float = 2.1
    model_weight: float = 2.2
    movement_weight: float = 0.025
    repetition_weight: float = 0.32
    contour_weight: float = 0.22
    min_relaxation: float = 0.13
    melodic_target: str = "climax"
    entropy_exponent: float = 4.0


@dataclass(frozen=True)
class StepTrace:
    bar: int
    beat: int
    previous_chord: str
    chord: str
    from_notes: tuple[int, ...]
    to_notes: tuple[int, ...]
    features: Features
    uncertainty: float
    multiplier: float
    transition_cost: float
    neural_nll: float


@dataclass
class Result:
    method: str
    frames: list[Frame]
    path: list[Voicing]
    steps: list[StepTrace]
    total_cost: float
    runtime_s: float
    reachable_counts: list[int]


class NoSolution(RuntimeError):
    pass


def contour_target(index: int, count: int, seed: int, profile: str = "arch") -> float:
    """A common time-varying melodic target, *shared* by all four systems.

    It is a simple preference for a contoured line, NOT a perception metric.
    """
    u = index / max(1, count-1)
    arch = 70 + 6 * (1 - cos(2*pi*u))/2 + 2.0 * (seed % 3 - 1)
    if profile == "climax" and index == count//2:
        # Intentional *predefined* one-event high-register climax target.
        # Same objective for all methods; tests ability to deviate from smoothness.
        return arch + 13
    return arch


def relaxation(method: str, step: int, frames: list[Frame], entropy: float, cfg: Config) -> float:
    if method in ('A','B'):
        return 1.0
    if method == 'C':
        # Pre-registered, model-independent heuristic: more flexibility around phrase middle.
        u = step / max(1, len(frames)-1)
        intensity = max(0.0, 1.0 - abs(u-0.5)/0.5)
        return 1.0 - 0.75*intensity
    if method == 'D':
        # Lower *output entropy* => lower soft penalties, bounded below.
        return cfg.min_relaxation + (1.0-cfg.min_relaxation)*(entropy**cfg.entropy_exponent)
    raise ValueError(method)


def run_one(chords: list[Chord], method: str, predictor, cfg: Config) -> Result:
    if method not in METHODS:
        raise ValueError(method)
    frames = make_frames(chords, cfg.meter, cfg.events_per_bar)
    t_start = perf_counter()
    lattice = [candidates(f.chord, cfg.cap, cfg.max_sa, cfg.max_at,
                          cfg.max_tb, cfg.root_bass, cfg.seed) for f in frames]
    if any(not pool for pool in lattice):
        raise NoSolution("A chord has no legal static voicings. Increase candidate cap or spacing limits.")

    # First layer: unary cost, same across systems.
    dp = {}
    for v in lattice[0]:
        centres = sum(abs(a-b) for a,b in zip(v.pitches, VOICE_CENTRES))
        cost = 0.03 * centres + cfg.contour_weight * abs(v.pitches[0] - contour_target(0,len(frames),cfg.seed,cfg.melodic_target))
        dp[v] = cost
    parents: list[dict[Voicing, tuple[Voicing, StepTrace]]] = []
    counts = [len(dp)]

    for step in range(1, len(frames)):
        current_frame, prior_frame = frames[step], frames[step-1]
        pool = lattice[step]
        allowed_sopranos = tuple(sorted(set(v.pitches[0] for v in pool)))
        next_dp: dict[Voicing, float] = {}
        next_parent: dict[Voicing, tuple[Voicing, StepTrace]] = {}
        for prev, prev_cost in dp.items():
            probabilities, entropy = predictor.distribution(
                prev.pitches[0], prior_frame.chord.root, current_frame.chord.root,
                current_frame.beat, allowed_sopranos)
            mult = relaxation(method, step, frames, entropy, cfg)
            for curr in pool:
                feat = transition_features(prev, curr, cfg.max_leap)
                if method == 'A' and (feat.parallel_fifths or feat.parallel_octaves or feat.oversized_leaps):
                    continue
                neural_nll = -log(max(1e-10, probabilities[curr.pitches[0]]))
                soft_penalty = mult * (
                    cfg.pf_weight*feat.parallel_fifths
                    + cfg.po_weight*feat.parallel_octaves
                    + cfg.leap_weight*feat.oversized_leaps)
                # Identical base objective for A/B/C/D. Time-varying target gives
                # all methods a fair reason to consider different legal voicings.
                base = (cfg.model_weight*neural_nll
                        + cfg.movement_weight*feat.melodic_distance
                        + cfg.repetition_weight*feat.repeated_voices
                        + cfg.contour_weight*abs(curr.pitches[0] - contour_target(step,len(frames),cfg.seed,cfg.melodic_target)))
                edge_cost = base + soft_penalty
                total = prev_cost + edge_cost
                if total < next_dp.get(curr, float('inf')):
                    next_dp[curr] = total
                    next_parent[curr] = (prev, StepTrace(
                        current_frame.bar, current_frame.beat + 1,
                        prior_frame.chord.name, current_frame.chord.name,
                        prev.pitches, curr.pitches, feat, entropy, mult,
                        edge_cost, neural_nll))
        if not next_dp:
            raise NoSolution(f"No reachable state at bar {current_frame.bar}, event {current_frame.beat+1} for system {method}. Try wider limits or a different progression.")
        dp = next_dp
        parents.append(next_parent)
        counts.append(len(dp))
    last = min(dp, key=dp.get)
    total_cost = dp[last]
    path = [last]
    traces = []
    for by_state in reversed(parents):
        previous, trace = by_state[path[-1]]
        traces.append(trace)
        path.append(previous)
    path.reverse()
    traces.reverse()
    return Result(method,frames,path,traces,total_cost,perf_counter()-t_start,counts)


def run_all(chords: list[Chord], predictor, cfg: Config) -> dict[str, Result | str]:
    return {m: _safe(chords, m, predictor, cfg) for m in METHODS}


def _safe(chords, method, predictor, cfg):
    try:
        return run_one(chords, method, predictor, cfg)
    except NoSolution as ex:
        return str(ex)
