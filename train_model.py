"""Train a small next-soprano Transformer for the Streamlit research prototype.

DEMO: generated synthetic transitions only; suitable for software verification,
NOT evidence that uncertainty reflects artistic confidence.

Usage:
    python train_model.py --demo --epochs 5
    python train_model.py --midi-dir ./licensed_midis --epochs 10 --output models/my_music.pt
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.nn import functional as F
from music_lab.neural import NoteTransformer, encode_context, DEFAULT_MODEL


def synthetic_data(count=2600, seed=19):
    rng = random.Random(seed)
    rows = []
    major = (0,2,4,5,7,9,11)
    minor = (0,2,3,5,7,8,10)
    for _ in range(count):
        tonic = rng.randrange(12)
        scale = rng.choice((major,minor))
        prev_root = (tonic+rng.choice(scale)) % 12
        next_root = (tonic+rng.choice(scale)) % 12
        quality = rng.choice(((0,4,7),(0,3,7),(0,4,7),(0,3,7)))
        allowed = [p for p in range(60,85) if p%12 in {(next_root+x)%12 for x in quality}]
        prev_sop = rng.randrange(63,80)
        # Data-specific melodic contour bias, plus exploration. No claim of real expressiveness.
        score = [abs(p-prev_sop)*0.7 + abs(p-72)*0.11 for p in allowed]
        if rng.random() < 0.17:
            target = rng.choice(allowed)
        else:
            pool = sorted(range(len(allowed)), key=lambda i:score[i])[:min(3,len(allowed))]
            target = allowed[rng.choice(pool)]
        rows.append((encode_context(prev_sop,prev_root,next_root,rng.randrange(4)),target))
    return rows


def midi_data(folder):
    try:
        import pretty_midi
    except ImportError as exc:
        raise RuntimeError('Install pretty_midi first') from exc
    paths = sorted(Path(folder).rglob('*.mid')) + sorted(Path(folder).rglob('*.midi'))
    rows = []
    read_files = 0
    for path in paths:
        try:
            midi = pretty_midi.PrettyMIDI(str(path))
            melodic = [inst for inst in midi.instruments if not inst.is_drum and len(inst.notes)>=8]
            if not melodic: continue
            # A simple highest-mean-pitch heuristic; imperfect melody extraction.
            soprano = max(melodic, key=lambda inst: np.mean([n.pitch for n in inst.notes]))
            melody = sorted((n for n in soprano.notes if 60 <= n.pitch <= 84), key=lambda n:n.start)
            if len(melody)<3: continue
            bpm = midi.estimate_tempo()
            if not (20 < bpm < 400): bpm=100.0
            def bass_pc_at(t):
                concurrent = [n.pitch for inst in melodic for n in inst.notes if n.start <= t < n.end]
                return (min(concurrent) % 12) if concurrent else 0
            for a,b in zip(melody,melody[1:]):
                p_root = bass_pc_at(a.start)
                c_root = bass_pc_at(b.start)
                beat = round(b.start*bpm/60)%4
                rows.append((encode_context(a.pitch,p_root,c_root,beat),b.pitch))
            read_files += 1
        except Exception as exc:
            print(f'Skipped {path.name}: {exc}')
    if not rows:
        raise ValueError('No suitable monophonic melody transitions in supplied MIDI files')
    return rows, read_files


def train(rows, output, epochs=5, batch_size=128, seed=19, source='synthetic'):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(4,torch.get_num_threads()))
    xs = np.array([x for x,y in rows],dtype='int64')
    ys = np.array([y for x,y in rows],dtype='int64')
    order = rng.permutation(len(xs))
    train_ids, val_ids = order[:max(1,int(.85*len(order)))],order[max(1,int(.85*len(order))):]
    xs = torch.tensor(xs); ys=torch.tensor(ys)
    model=NoteTransformer()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.004)
    start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        perm = train_ids[rng.permutation(len(train_ids))]
        total=0
        for idx in np.array_split(perm,max(1,(len(perm)+batch_size-1)//batch_size)):
            batch=torch.tensor(idx,dtype=torch.long)
            logits=model(xs[batch]); loss=F.cross_entropy(logits,ys[batch])
            optimizer.zero_grad();loss.backward();optimizer.step()
            total+=float(loss)*len(idx)
        model.eval()
        with torch.no_grad():
            vloss=float(F.cross_entropy(model(xs[val_ids]),ys[val_ids])) if len(val_ids) else float('nan')
        print(f'Epoch {epoch+1}/{epochs} train NLL={total/len(train_ids):.3f} validation NLL={vloss:.3f}')
    output=Path(output); output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(model.state_dict(),output)
    meta={'source':source,'examples':len(rows),'epochs':epochs,'seed':seed,
          'validation_nll':vloss,'seconds':round(time.perf_counter()-start,2),
          'warning':'Synthetic model demonstrates architecture only; not validated artistic confidence.' if source=='synthetic' else 'Real MIDI melody/root extraction is heuristic. Validate annotations and splits.'}
    output.with_suffix('.json').write_text(json.dumps(meta,indent=2))
    print('Saved',output)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    g=p.add_mutually_exclusive_group()
    g.add_argument('--demo',action='store_true')
    g.add_argument('--midi-dir',type=str)
    p.add_argument('--output',default=str(DEFAULT_MODEL))
    p.add_argument('--epochs',type=int,default=5)
    p.add_argument('--samples',type=int,default=2600)
    p.add_argument('--seed',type=int,default=19)
    a=p.parse_args()
    if a.midi_dir:
        examples,files=midi_data(a.midi_dir); label=f'midi:{files}files'
    else:
        examples=synthetic_data(a.samples,a.seed);label='synthetic'
    train(examples,a.output,a.epochs,seed=a.seed,source=label)
