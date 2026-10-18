"""Eight-voice Bach-derived *problem instance* experiments.

The supplied JSONs contain starting pitches and a sequence of note anchors and
quarters-length durations, NOT synchronized ground-truth Bach SATB notes.
This module therefore implements anchor-conditioned experimental multi-voice
search. Its output is not a Bach transcription or a performance model.

All systems share one deterministic, truncated lattice; the core algorithm is
Viterbi DP on that lattice and is NOT an FPT algorithm.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from math import log
from io import BytesIO
from pathlib import Path
from time import perf_counter
import json
import random
import hashlib
import pretty_midi
import pandas as pd
from .core import note_name
from .search import METHODS

@dataclass(frozen=True)
class BachProblem:
    name: str
    anchors: tuple[int, ...]    # MIDI numbers from chord_roots, not complete chords
    durations: tuple[float, ...]
    starts: tuple[int, ...]
    max_step: int
    max_k: int
    metadata: dict
    sha256: str

    @property
    def voices(self) -> int:
        return len(self.starts)

    @property
    def events(self) -> int:
        return len(self.anchors)


def read_problem(raw: bytes | str | Path) -> BachProblem:
    """Validate an uploaded problem. Strings represent JSON text or a disk path."""
    if isinstance(raw, Path):
        raw = raw.read_bytes()
    if isinstance(raw, str):
        if raw.lstrip().startswith('{'):
            raw = raw.encode('utf8')
        else:
            raw = Path(raw).read_bytes()
    d = json.loads(raw)
    name = str(d['problem_id'])
    if not name.startswith('chorale_'):
        raise ValueError('Expected a chorale problem_id')
    roots = d['chord_roots']
    ds = d['chord_durations']
    starts = tuple(int(x) for x in d['starting_pitches'])
    if not (2 <= len(roots) == len(ds) <= 500):
        raise ValueError('Require 2–500 anchors and one duration per anchor')
    if not (len(starts) == int(d['voices']) == 8):
        raise ValueError('Eight-voice mode requires eight starting pitches')
    if not all(0 <= p <= 127 for p in starts):
        raise ValueError('Starting pitch outside MIDI range')
    if not all(starts[i] >= starts[i+1] for i in range(7)):
        raise ValueError('Starting pitches must be ordered high-to-low')
    durations = tuple(float(x) for x in ds)
    if not all(0 < x <= 16 for x in durations):
        raise ValueError('Durations must be positive and <=16 quarter notes')
    anchors = tuple(int(pretty_midi.note_name_to_number(x.replace('-', 'b'))) for x in roots)
    limits = d['constraints']
    max_step = int(limits['max_melodic_step'])
    max_k = int(limits['max_k'])
    if not 1 <= max_step <= 12 or not 0 <= max_k <= 7:
        raise ValueError('Bad max_melodic_step or max_k')
    return BachProblem(name,anchors,durations,starts,max_step,max_k,d.get('metadata',{}),hashlib.sha256(raw).hexdigest())


@dataclass(frozen=True)
class MultiVoicing:
    pitches: tuple[int, ...]


def crossing_stats(pitches: tuple[int, ...]) -> tuple[int,int]:
    """Pair inversions and largest voice rank displacement relative to source order."""
    n=len(pitches)
    order=sorted(range(n),key=lambda i:(-pitches[i],i))
    ranks={voice:rank for rank,voice in enumerate(order)}
    crossings=sum(pitches[i]<pitches[j] for i in range(n) for j in range(i+1,n))
    return crossings,max(abs(ranks[i]-i) for i in range(n))


def features(prev: MultiVoicing,curr: MultiVoicing,limit: int) -> tuple[int,int,int,int,float,int]:
    x,y=prev.pitches,curr.pitches
    fifths=octaves=0
    for i,j in combinations(range(len(x)),2):
        d1=y[i]-x[i]; d2=y[j]-x[j]
        if d1 and d2 and (d1>0)==(d2>0):
            pi=abs(x[i]-x[j])%12; ci=abs(y[i]-y[j])%12
            fifths += pi==ci==7
            octaves += pi==ci==0
    motion=[abs(a-b) for a,b in zip(x,y)]
    over=sum(d>limit for d in motion)
    return fifths,octaves,over,sum(motion),sum(motion)/len(motion),sum(a==b for a,b in zip(x,y))


def pc_distance(pc1: int,pc2:int)->int:
    d=abs((pc1%12)-(pc2%12));return min(d,12-d)


@dataclass(frozen=True)
class BachConfig:
    cap: int = 24
    seed: int = 7
    k: int = 3
    max_displacement: int = 5
    pf_weight: float = 5.0
    po_weight: float = 6.0
    leap_weight: float = 3.0
    note_weight: float = 0.7
    anchor_weight: float = 1.4
    movement_weight: float = .065
    repetition_weight: float = .28
    min_relaxation: float = .15
    entropy_exponent: float = 3.0


def _valid_static(notes: tuple[int,...],starts:tuple[int,...],cfg:BachConfig)->bool:
    return (all(0<=v<=127 and abs(v-starts[i])<=cfg.max_displacement for i,v in enumerate(notes))
            and crossing_stats(notes)[1]<=cfg.k)


def make_lattice(problem:BachProblem,cfg:BachConfig) -> list[tuple[MultiVoicing,...]]:
    """Fixed per-time candidate states; the same lattice is used in A/B/C/D.

    Candidates are constructed from perturbations around the original eight
    starting pitches, not from inaccessible historical target harmonizations.
    """
    rng=random.Random(cfg.seed)
    starts=problem.starts
    out=[(MultiVoicing(starts),)]
    for idx,anchor in enumerate(problem.anchors[1:],start=1):
        options={starts}
        # Deterministic local changes plus anchor-oriented soprano pitches.
        # No hard harmonic restriction is inferred from single-note anchors.
        candidates=[]
        for j in range(8):
            for delta in (-3,-2,-1,1,2,3):
                s=list(starts);s[j]+=delta
                if _valid_static(tuple(s),starts,cfg):candidates.append(tuple(s))
        for _ in range(3*cfg.cap):
            s=list(starts)
            which=rng.sample(range(8),rng.choice((1,2,3,4)))
            for j in which:s[j]+=rng.choice((-4,-3,-2,-1,1,2,3,4))
            if _valid_static(tuple(s),starts,cfg):candidates.append(tuple(s))
        # Always use an anchor-aware ranking to choose a shared time-local lattice.
        r=anchor%12
        candidates=sorted(set(candidates),key=lambda s:(
            1.4*pc_distance(s[0]%12,r)+.18*sum(abs(a-b) for a,b in zip(s,starts)),
            sum(abs(a-b) for a,b in zip(s,starts)),s))
        # Every event includes the unchanged seed state, guaranteeing at least
        # a feasible A path and keeping failure detection meaningful.
        selected=[starts]
        for s in candidates:
            if s not in selected:
                selected.append(s)
            if len(selected)>=cfg.cap:break
        out.append(tuple(MultiVoicing(s) for s in selected))
    return out


def weight_multiplier(method:str,t:int,count:int,u:float,cfg:BachConfig)->float:
    if method in ('A','B'):return 1.0
    if method=='C':
        x=t/max(1,count-1)
        return 1.0-.75*max(0.0,1.0-abs(x-.5)/.5)
    if method=='D':
        return cfg.min_relaxation+(1.0-cfg.min_relaxation)*(u**cfg.entropy_exponent)
    raise ValueError(method)


@dataclass
class BachResult:
    method:str
    problem:BachProblem
    states:list[MultiVoicing]
    traces:list[dict]
    score:float
    seconds:float
    reachable:list[int]


def run_one(problem:BachProblem,method:str,predictor,cfg:BachConfig,lattice=None)->BachResult:
    if method not in METHODS:raise ValueError(method)
    if cfg.k > problem.max_k:raise ValueError(f'k cannot exceed dataset max_k={problem.max_k}')
    t0=perf_counter()
    lattice=lattice if lattice is not None else make_lattice(problem,cfg)
    dp={state:0.0 for state in lattice[0]}
    links=[]; counts=[len(dp)]
    for t in range(1,problem.events):
        pool=lattice[t]
        allowed=tuple(sorted({v.pitches[0] for v in pool}))
        best={};parent={}
        prev_pc=problem.anchors[t-1]%12;anchor_pc=problem.anchors[t]%12
        for prev,old in dp.items():
            probs,u=predictor.distribution(prev.pitches[0],prev_pc,anchor_pc,t%4,allowed)
            multiplier=weight_multiplier(method,t,problem.events,u,cfg)
            for nxt in pool:
                f=features(prev,nxt,problem.max_step)
                fifths,octaves,overs,motion,_,repeats=f
                if method=='A' and (fifths or octaves or overs):continue
                # Predictor was synthetically trained on soprano/chord contexts;
                # anchor PCs are proxies here and cannot establish Bach prediction.
                nll=-log(max(1e-10,probs[nxt.pitches[0]]))
                anchor_cost=pc_distance(nxt.pitches[0],anchor_pc)
                center_cost=sum(abs(a-b) for a,b in zip(nxt.pitches,problem.starts))
                base=(cfg.note_weight*nll + cfg.anchor_weight*anchor_cost
                      +cfg.movement_weight*motion +cfg.repetition_weight*repeats
                      +.025*center_cost)
                soft=multiplier*(cfg.pf_weight*fifths+cfg.po_weight*octaves+cfg.leap_weight*overs)
                edge=base+soft
                cost=old+edge
                if cost<best.get(nxt,float('inf')):
                    best[nxt]=cost
                    parent[nxt]=(prev,{
                        'Event':t+1,'Elapsed quarter-note beats':round(sum(problem.durations[:t]),3),
                        'Anchor pitch':note_name(problem.anchors[t]),
                        'Parallel fifths':fifths,'Parallel octaves':octaves,
                        'Over-limit voice leaps':overs,'Total voice motion (semitones)':motion,
                        'Predictive entropy (0–1)':u,'Penalty multiplier (0–1)':multiplier,
                        'Local transition cost (a.u.)':edge})
        if not best:raise RuntimeError(f'No path reaches event {t+1} for method {method}')
        dp=best;links.append(parent);counts.append(len(dp))
    last=min(dp,key=dp.get);score=dp[last]
    path=[last];traces=[]
    for links_at_t in reversed(links):
        before,trace=links_at_t[path[-1]];path.append(before);traces.append(trace)
    path.reverse();traces.reverse()
    return BachResult(method,problem,path,traces,score,perf_counter()-t0,counts)


def run_all(problem:BachProblem,predictor,cfg:BachConfig)->dict[str,BachResult]:
    lattice=make_lattice(problem,cfg)
    return {m:run_one(problem,m,predictor,cfg,lattice=lattice) for m in METHODS}


def bach_metrics(r:BachResult)->dict:
    traces=r.traces
    n=max(1,len(traces))
    pcs=[s.pitches[0]%12 for s in r.states]
    return {
        'Chorale':r.problem.name,'System':r.method,
        'Events (count)':r.problem.events,'Voices (count)':r.problem.voices,
        'Parallel fifths (count)':sum(t['Parallel fifths'] for t in traces),
        'Parallel octaves (count)':sum(t['Parallel octaves'] for t in traces),
        'Leaps above dataset threshold (count)':sum(t['Over-limit voice leaps'] for t in traces),
        'Mean total movement (semitones/event)':round(sum(t['Total voice motion (semitones)'] for t in traces)/n,3),
        'Anchor match (soprano pitch-class %)':round(100*sum(pc==root%12 for pc,root in zip(pcs,r.problem.anchors))/len(pcs),2),
        'Distinct soprano pitches (count)':len({s.pitches[0] for s in r.states}),
        'Max observed voice rank displacement':max(crossing_stats(v.pitches)[1] for v in r.states),
        'Runtime (seconds)':round(r.seconds,3),
        'Final objective (a.u.)':round(r.score,3),
    }


def bach_notes(r:BachResult)->pd.DataFrame:
    rows=[];position=0.0
    for t,(v,d,anchor) in enumerate(zip(r.states,r.problem.durations,r.problem.anchors)):
        row={'Event':t+1,'Start (quarter-note beats)':round(position,3),'Duration (quarter-note beats)':d,'Anchor (input)':note_name(anchor)}
        for i,p in enumerate(v.pitches):row[f'Voice {i+1} (MIDI)']=p;row[f'Voice {i+1} (note)']=note_name(p)
        rows.append(row);position+=d
    return pd.DataFrame(rows)


def bach_midi(r:BachResult,tempo:float=90.0)->bytes:
    pm=pretty_midi.PrettyMIDI(initial_tempo=tempo)
    seconds_per_beat=60.0/tempo
    offsets=[];elapsed=0.0
    for dur in r.problem.durations:offsets.append(elapsed);elapsed+=dur
    for j in range(r.problem.voices):
        inst=pretty_midi.Instrument(program=48+j%8,name=f'Experimental voice {j+1}')
        last=None;start=0.0
        for t,v in enumerate(r.states):
            pitch=v.pitches[j];at=offsets[t]*seconds_per_beat
            if last is None:last=pitch;start=at
            elif pitch!=last:
                inst.notes.append(pretty_midi.Note(velocity=78,pitch=last,start=start,end=at))
                start=at;last=pitch
        if last is not None:
            inst.notes.append(pretty_midi.Note(velocity=78,pitch=last,start=start,end=elapsed*seconds_per_beat))
        pm.instruments.append(inst)
    # pretty_midi's writer inserts a default 4/4 even without a score meter.
    # Strip that implicit metadata so the export does not pretend the supplied
    # JSON identified the source chorale's time signature.
    import mido
    bio=BytesIO();pm.write(bio)
    mf=mido.MidiFile(file=BytesIO(bio.getvalue()))
    for track in mf.tracks:
        track[:] = [message for message in track if message.type != 'time_signature']
    clean=BytesIO();mf.save(file=clean)
    return clean.getvalue()


def bach_validation(problems:list[BachProblem])->pd.DataFrame:
    return pd.DataFrame([{
        'File':p.name,'Number of events':p.events,'Voices':p.voices,
        'Quarter-note duration (total)':round(sum(p.durations),2),
        'Original source voices (metadata)':p.metadata.get('original_voice_count','unknown'),
        'Max leap threshold (semitones)':p.max_step,
        'Allowed max k':p.max_k,'SHA256 (prefix)':p.sha256[:12]
    } for p in problems])
