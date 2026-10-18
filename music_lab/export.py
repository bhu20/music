"""MIDI and CSV exports."""
from __future__ import annotations
from io import BytesIO
import pretty_midi
import pandas as pd
from .search import Result
from .metrics import notes_dataframe, trace_dataframe


def midi_bytes(result: Result, tempo: float = 88) -> bytes:
    if tempo <= 0: raise ValueError('Tempo must be positive')
    pm = pretty_midi.PrettyMIDI(initial_tempo=float(tempo))
    pm.time_signature_changes.append(pretty_midi.TimeSignature(result.frames[0].meter,4,0))
    instruments = [pretty_midi.Instrument(program=n, name=label) for n,label in zip(
        (52,53,54,42),('Soprano','Alto','Tenor','Bass'))]
    beats_per_event = result.frames[0].meter / (len(result.frames)//len({f.bar for f in result.frames}))
    event_duration = 60.0/tempo * beats_per_event
    for j, instrument in enumerate(instruments):
        # Merge repeated pitches into a held MIDI note. No microtiming or expressive articulation.
        current_pitch = None
        start_index = 0
        for i in range(len(result.path)+1):
            p = result.path[i].pitches[j] if i < len(result.path) else None
            if p != current_pitch:
                if current_pitch is not None:
                    instrument.notes.append(pretty_midi.Note(
                        velocity=88 if j==0 else 75, pitch=current_pitch,
                        start=start_index*event_duration, end=i*event_duration))
                current_pitch = p
                start_index = i
        pm.instruments.append(instrument)
    bio = BytesIO()
    pm.write(bio)
    return bio.getvalue()


def summary_csv(results):
    from .metrics import summarize
    return pd.DataFrame([summarize(r) for r in results.values() if isinstance(r, Result)]).to_csv(index=False).encode()


def preview_wav(result: Result, tempo: float = 100.0, sample_rate: int = 22050) -> bytes:
    """Simple harmonic synthesizer to preview voices without soundfonts.

    This is intentionally a plain reference rendering, NOT an audio-quality score.
    """
    import wave
    from math import pi
    import numpy as np
    midi = midi_bytes(result, tempo)
    pm = pretty_midi.PrettyMIDI(BytesIO(midi))
    duration = pm.get_end_time() + .1
    n=int(duration*sample_rate)
    audio=np.zeros(n,dtype=np.float64)
    for channel, inst in enumerate(pm.instruments):
        for note in inst.notes:
            left=max(0,int(note.start*sample_rate));right=min(n,int(note.end*sample_rate))
            if right<=left: continue
            samples=np.arange(right-left,dtype=np.float64)/sample_rate
            hz=pretty_midi.note_number_to_hz(note.pitch)
            partial=(np.sin(2*pi*hz*samples)+.17*np.sin(4*pi*hz*samples)+.07*np.sin(6*pi*hz*samples))
            attack=np.minimum(1, samples/.025)
            release=np.minimum(1,(len(samples)-np.arange(len(samples)))/(sample_rate*.08))
            audio[left:right]+= (.20 if channel==0 else .13) * partial * attack * release
    amp=max(1.0,float(np.max(np.abs(audio)))/.90)
    pcm=(np.clip(audio/amp,-1,1)*32767).astype('<i2')
    bio=BytesIO()
    with wave.open(bio,'wb') as wav:
        wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return bio.getvalue()
