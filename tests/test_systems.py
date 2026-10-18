import io
import pretty_midi
import pytest
from music_lab.core import parse_progression,static_errors,transition_features
from music_lab.search import Config,Result,run_all,run_one,relaxation
from music_lab.neural import Predictor,DEFAULT_MODEL,HeuristicPredictor
from music_lab.metrics import summarize,trace_dataframe,notes_dataframe
from music_lab.export import midi_bytes,preview_wav

@pytest.fixture(scope='module')
def compare():
    cfg=Config(cap=55,meter=4,events_per_bar=2,max_leap=6,
               contour_weight=2.2,melodic_target='climax',model_weight=1.8,leap_weight=3.0)
    return run_all(parse_progression('C G Am F G C'),Predictor(),cfg),cfg


def test_demo_model_checkpoint_exists():
    assert DEFAULT_MODEL.exists()


def test_all_systems_run_and_share_hard_invariants(compare):
    results,cfg=compare
    assert set(results)=={'A','B','C','D'}
    for name,r in results.items():
        assert isinstance(r,Result),f'{name}: {r}'
        assert len(r.path)==12
        assert len(r.steps)==11
        for frame,v in zip(r.frames,r.path):
            assert not static_errors(v,frame.chord)
        assert len(r.reachable_counts)==len(r.path)
        assert all(c>0 for c in r.reachable_counts)


def test_system_a_is_strict(compare):
    results,cfg=compare
    for a,b in zip(results['A'].path,results['A'].path[1:]):
        f=transition_features(a,b,cfg.max_leap)
        assert f.parallel_fifths==f.parallel_octaves==f.oversized_leaps==0


def test_system_c_changes_penalties_in_phrase(compare):
    results,cfg=compare
    c=results['C']
    weights=[t.multiplier for t in c.steps]
    assert min(weights) < max(weights)
    assert max(weights)<=1.0
    assert all(0.0<=t.uncertainty<=1 for t in results['D'].steps)
    assert any(t.multiplier<1 for t in results['D'].steps)


def test_common_climax_stress_case_produces_reproducible_difference(compare):
    results,cfg=compare
    # A vs C diverge in this frozen stress-test setup. No claim C is better.
    assert tuple(x.pitches for x in results['A'].path) != tuple(x.pitches for x in results['C'].path)
    second=run_one(parse_progression('C G Am F G C'),'C',Predictor(),cfg)
    assert [x.pitches for x in second.path]==[x.pitches for x in results['C'].path]


def test_metric_tables_and_audio_exports(compare):
    results,_=compare
    r=results['A']
    assert summarize(r)['Total violations (count)']==0
    assert len(notes_dataframe(r))==len(r.path)
    assert len(trace_dataframe(r))==len(r.steps)
    payload=midi_bytes(r,tempo=99)
    assert payload[:4]==b'MThd'
    music=pretty_midi.PrettyMIDI(io.BytesIO(payload))
    assert len(music.instruments)==4
    assert music.time_signature_changes[0].numerator==4
    wav=preview_wav(r,99)
    assert wav[:4]==b'RIFF'


def test_time_signature_in_midi():
    r=run_one(parse_progression('C G'),'A',HeuristicPredictor(),Config(cap=35,meter=2,events_per_bar=2))
    pm=pretty_midi.PrettyMIDI(io.BytesIO(midi_bytes(r)))
    assert pm.time_signature_changes[0].numerator==2
    assert abs(pm.get_end_time()-4*60/88)<.05
