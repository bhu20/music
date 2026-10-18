import pytest
from music_lab.core import Chord, Voicing, candidates, static_errors, transition_features, parse_progression, make_frames


def test_parse_major_minor_seventh_and_bad_name():
    chords=parse_progression('C G7 Am Fmaj7')
    assert chords[0].pitch_classes == frozenset({0,4,7})
    assert chords[1].pitch_classes == frozenset({7,11,2,5})
    assert chords[2].pitch_classes == frozenset({9,0,4})
    assert len(chords[3].pitch_classes)==4
    with pytest.raises(ValueError): parse_progression('C Xx')


def test_candidate_static_legality_and_cap():
    chord=parse_progression('C G')[0]
    pool=candidates(chord, 75)
    assert 1<=len(pool)<=75
    for v in pool:
        assert not static_errors(v,chord)


def test_static_errors_detect_missing_third_and_spacing():
    c=parse_progression('C G')[0]
    power=Voicing((79,67,55,48))  # G4,G4,G3,C3, misses E, also equals? defined broad
    errors=static_errors(power,c)
    assert 'incomplete chord' in errors
    spread=Voicing((84,64,55,48))
    assert any('spacing' in e for e in static_errors(spread,c))


def test_parallel_fifth_with_motion_and_nonparallel_without_motion():
    # (C5, F4) => (D5, G4): both perfect fifths, both go upward.
    a=Voicing((72,65,60,48))
    b=Voicing((74,67,62,50))
    assert transition_features(a,b).parallel_fifths>=1
    stationary=Voicing((72,65,60,48))
    assert transition_features(a,stationary).parallel_fifths==0


def test_parallel_octave_and_leap():
    a=Voicing((72,60,55,48))
    b=Voicing((74,62,55,36))
    f=transition_features(a,b,max_leap=7)
    assert f.parallel_octaves >= 1
    assert f.oversized_leaps==1


def test_events_meter():
    chords=parse_progression('C G Am')
    frames=make_frames(chords,meter=3,beats_per_bar=3)
    assert len(frames)==9
    assert frames[3].bar==2 and frames[3].beat==0
