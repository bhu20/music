"""Run all supplied Bach-derived JSON experiments headlessly and export outputs.

python run_bach_demo.py --output examples/bach_results
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from music_lab.bach import read_problem, BachConfig,run_all,bach_metrics,bach_notes,bach_midi
from music_lab.neural import Predictor,HeuristicPredictor


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--folder',type=Path,default=Path('data/bach'))
    ap.add_argument('--output',type=Path,default=Path('examples/bach_results'))
    ap.add_argument('--cap',type=int,default=24)
    ap.add_argument('--k',type=int,default=3)
    ap.add_argument('--seed',type=int,default=7)
    ap.add_argument('--debug-heuristic',action='store_true')
    a=ap.parse_args()
    a.output.mkdir(exist_ok=True,parents=True)
    cfg=BachConfig(cap=a.cap,seed=a.seed,k=a.k)
    predictor=HeuristicPredictor() if a.debug_heuristic else Predictor()
    records=[]
    all_runs={}
    for path in sorted(a.folder.glob('chorale_*.json')):
        problem=read_problem(path)
        if a.k>problem.max_k:raise ValueError('Requested k exceeds dataset limit')
        print('Running',problem.name,'events',problem.events)
        outputs=run_all(problem,predictor,cfg)
        all_runs[problem.name]=outputs
        for name,r in outputs.items():
            records.append(bach_metrics(r))
            (a.output/f'{problem.name}_{name}.mid').write_bytes(bach_midi(r))
            bach_notes(r).to_csv(a.output/f'{problem.name}_{name}_notes.csv',index=False)
            pd.DataFrame(r.traces).to_csv(a.output/f'{problem.name}_{name}_audit.csv',index=False)
    df=pd.DataFrame(records)
    df.to_csv(a.output/'bach_abcd_metrics.csv',index=False)
    print(df[['Chorale','System','Parallel fifths (count)','Parallel octaves (count)',
              'Leaps above dataset threshold (count)','Anchor match (soprano pitch-class %)']].to_string(index=False))
    print('Outputs:',a.output)
    # Static preview designed for papers/quick inspection with explicit labels.
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,2,figsize=(12,7),sharex=True,sharey=True)
        rset=next(iter(all_runs.values()))
        for ax,(m,r) in zip(axes.ravel(),rset.items()):
            times=[0.0]
            for dur in r.problem.durations[:-1]:times.append(times[-1]+dur)
            for j in range(8):
                ax.plot(times,[s.pitches[j] for s in r.states],lw=1.05,label=f'Voice {j+1}')
            ax.set_title(f'System {m}')
            ax.set_xlabel('Elapsed quarter-note beats')
            ax.set_ylabel('Pitch (MIDI number)')
            ax.grid(alpha=.18)
        handles,labels=axes[0][0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='upper center',ncol=8,bbox_to_anchor=(.5,1.03),fontsize=8)
        fig.suptitle(f'Eight-voice trajectories | {next(iter(all_runs))}',y=1.08)
        fig.tight_layout()
        fig.savefig(a.output/'bach_abcd_8voice_preview.png',dpi=155,bbox_inches='tight')
        plt.close(fig)
    except ImportError:
        print('matplotlib not installed; no static preview created. Streamlit plots remain available.')


if __name__=='__main__':main()
