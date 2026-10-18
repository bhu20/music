"""Real supplied Bach-derived JSON integration checks. No original scores assumed."""
from pathlib import Path
import io
import pytest
import pretty_midi
from music_lab.bach import (read_problem,BachConfig,make_lattice,run_all,features,
                            crossing_stats,bach_metrics,bach_notes,bach_midi)
from music_lab.neural import HeuristicPredictor

DATA=Path(__file__).resolve().parents[1]/'data'/'bach'
FILES=sorted(DATA.glob('chorale_*.json'))

@pytest.mark.parametrize('path',FILES)
def test_dataset_schema(path):
    d=read_problem(path)
    assert d.events>50
    assert d.voices==8
    assert len(d.durations)==d.events
    assert d.max_step==2
    assert d.max_k==3
    assert d.metadata['original_voice_count']==4


def test_lattice_is_held_constant_across_methods():
    problem=read_problem(FILES[0])
    cfg=BachConfig(cap=18,seed=9)
    assert make_lattice(problem,cfg)==make_lattice(problem,cfg)
    assert len(make_lattice(problem,cfg))==problem.events

@pytest.mark.parametrize('path',FILES)
def test_all_methods_on_real_problem_files(path):
    d=read_problem(path);cfg=BachConfig(cap=20,seed=7,k=3)
    runs=run_all(d,HeuristicPredictor(),cfg)
    assert set(runs)=={'A','B','C','D'}
    for key,r in runs.items():
        assert len(r.states)==d.events
        assert len(r.traces)==d.events-1
        assert all(c>0 for c in r.reachable)
        assert all(crossing_stats(s.pitches)[1]<=cfg.k for s in r.states)
        assert len(bach_notes(r))==d.events
        assert 0<=bach_metrics(r)['Anchor match (soprano pitch-class %)']<=100
        assert all(0<=x['Predictive entropy (0–1)']<=1 for x in r.traces)
    for prev,next in zip(runs['A'].states,runs['A'].states[1:]):
        f=features(prev,next,d.max_step)
        assert f[0]==f[1]==f[2]==0


def test_dataset_specific_durations_are_used_in_midi():
    d=read_problem(FILES[1]);r=run_all(d,HeuristicPredictor(),BachConfig(cap=12))['A']
    pm=pretty_midi.PrettyMIDI(io.BytesIO(bach_midi(r,tempo=120)))
    assert len(pm.instruments)==8
    assert all(len(inst.notes)>=1 for inst in pm.instruments)
    assert abs(pm.get_end_time()-sum(d.durations)*.5)<.03
    # No made-up time signature (the JSON does not specify one)
    assert not pm.time_signature_changes


def test_strict_k_zero():
    d=read_problem(FILES[2]);r=run_all(d,HeuristicPredictor(),BachConfig(cap=12,k=0))['A']
    assert all(crossing_stats(s.pitches)[1]==0 for s in r.states)


def test_invalid_dataset_rejected():
    with pytest.raises((ValueError,KeyError)):
        read_problem('{"problem_id":"chorale_fake","voices":8,"chord_roots":[],"chord_durations":[],"starting_pitches":[1,2,3,4,5,6,7,8],"constraints":{}}')
