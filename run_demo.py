"""Headless reproducibility example: generate all four MIDI sequences and CSV reports."""
from pathlib import Path
import json
import hashlib
import pandas as pd
from music_lab.core import parse_progression
from music_lab.neural import Predictor,DEFAULT_MODEL
from music_lab.search import Config,run_all,Result
from music_lab.metrics import summarize,notes_dataframe,trace_dataframe
from music_lab.export import midi_bytes


def main():
    target=Path(__file__).parent/'examples'
    target.mkdir(exist_ok=True)
    progression='C G Am F G C'
    cfg=Config(cap=55,max_leap=6,events_per_bar=2,meter=4,
               melodic_target='climax',contour_weight=2.2,leap_weight=3.0,model_weight=1.8)
    results=run_all(parse_progression(progression),Predictor(),cfg)
    metrics=[]
    for method,r in results.items():
        if not isinstance(r,Result):
            print(f'{method}: {r}')
            continue
        (target/f'system_{method}.mid').write_bytes(midi_bytes(r,tempo=94))
        notes_dataframe(r).to_csv(target/f'system_{method}_notes.csv',index=False)
        trace_dataframe(r).to_csv(target/f'system_{method}_audit.csv',index=False)
        metrics.append(summarize(r))
    pd.DataFrame(metrics).to_csv(target/'metrics.csv',index=False)
    meta={'progression':progression,'model_sha256':hashlib.sha256(DEFAULT_MODEL.read_bytes()).hexdigest(),
          'model_provenance':'synthetic demonstration only',
          'config':vars(cfg),
          'note':'Results depend on truncated candidate lattice; do not infer musical quality from violation counts.'}
    (target/'experiment.json').write_text(json.dumps(meta,indent=2))
    print(pd.DataFrame(metrics).to_string(index=False))
    print('Saved example artifacts to',target)


if __name__=='__main__':
    main()
