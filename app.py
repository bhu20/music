"""Streamlit interactive A/B/C/D music search research workbench."""
import json
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from music_lab.core import PRESETS, parse_progression, note_name
from music_lab.neural import Predictor, HeuristicPredictor, DEFAULT_MODEL
from music_lab.search import METHODS, Config, Result, run_all
from music_lab.metrics import summarize, notes_dataframe, trace_dataframe
from music_lab.export import midi_bytes, preview_wav, summary_csv

st.set_page_config(page_title="Music Constraint Research Lab", page_icon="🎼", layout="wide")
st.title("Music Constraint Research Lab")
st.caption("Controlled A/B/C/D experiment | SATB symbolic generation | dynamic programming on a shared pruned lattice")

@st.cache_resource(show_spinner=False)
def get_predictor(which: str, path: str):
    if which == "Heuristic (debug only)":
        return HeuristicPredictor()
    return Predictor(path)

with st.sidebar:
    st.header("Experiment settings")
    preset = st.selectbox("Progression preset", list(PRESETS), index=0)
    progression = st.text_input("Chord progression", value=PRESETS[preset], help="Example: C G Am F G C. Chords: major, m, dim, aug, 7, maj7, m7")
    meter_str = st.selectbox("Time signature", ["2/4","3/4","4/4"], index=2)
    meter = int(meter_str[0])
    beat_events = st.slider("Note events per bar", 1, meter, min(2,meter), 1,
                            help="Each voice can change notes only at these equally spaced positions. Notes may be sustained by repetition.")
    melodic_target = st.selectbox("Common melodic objective", ["climax", "arch"],
                                  format_func=lambda x: "Single-event dramatic climax (contrast test)" if x=="climax" else "Smooth arch")
    model_choice = st.selectbox("Base note predictor", ["Bundled synthetic Transformer", "Heuristic (debug only)", "My trained Transformer"], index=0)
    custom_path = st.text_input("My local model path", "models/my_music.pt") if model_choice=="My trained Transformer" else ""
    st.markdown("**Search limits**")
    cap = st.slider("Candidate voicings per chord", 20, 150, 55, 5)
    max_leap = st.slider("Oversized leap threshold (semitones)", 3, 14, 6)
    sa = st.slider("Maximum S–A spacing (semitones)", 8, 16, 12)
    at = st.slider("Maximum A–T spacing (semitones)", 8, 16, 12)
    tb = st.slider("Maximum T–B spacing (semitones)", 12, 24, 19)
    root_bass = st.checkbox("Require chord root in bass", False)
    seed = st.number_input("Deterministic candidate seed", min_value=0, max_value=999, value=5)
    st.markdown("**Penalty coefficients**")
    pf = st.slider("Parallel fifth penalty", 0.0, 12.0, 3.8, .2)
    po = st.slider("Parallel octave penalty", 0.0, 12.0, 5.0, .2)
    le = st.slider("Oversized leap penalty", 0.0, 8.0, 3.0, .2)
    entropy_exponent = st.slider("D: entropy calibration exponent", 1.0, 8.0, 4.0, .5, help="Higher exponent produces stronger relaxation for a given predictive entropy. Pre-register this setting before experiments.")
    model_weight = st.slider("Learned/heuristic preference weight", 0.0, 5.0, 1.8, .2)
    contour_weight = st.slider("Melodic contour objective strength", 0.0, 4.0, 2.2, .1)
    tempo = st.slider("Playback tempo (BPM)", 50, 180, 94)
    run_clicked = st.button("Run systems A, B, C, D", type="primary", use_container_width=True)

st.info("**Scope:** fixed harmony with synchronized SATB note events on a regular rhythmic grid. This prototype does not generate free rhythms, rests, passing tones, lyrics, or long-range phrase structure. 'D' uses **predictive output entropy**, not attention entropy. The bundled model is trained on synthetic data only.")

if run_clicked:
    try:
        chords=parse_progression(progression)
        config=Config(cap=cap,meter=meter,events_per_bar=beat_events,max_leap=max_leap,
                      max_sa=sa,max_at=at,max_tb=tb,root_bass=root_bass,seed=int(seed),melodic_target=melodic_target,entropy_exponent=entropy_exponent,
                      pf_weight=pf,po_weight=po,leap_weight=le,model_weight=model_weight,contour_weight=contour_weight)
        if model_choice=="My trained Transformer":
            path=str(Path(custom_path).expanduser().resolve())
        else:
            path=str(DEFAULT_MODEL)
        predictor=get_predictor(model_choice,path)
        with st.spinner("Finding optimal paths through the same voicing lattice for all four systems..."):
            results=run_all(chords,predictor,config)
        st.session_state["exp"]=(results,tempo,model_choice, progression,config)
    except Exception as exc:
        st.error(f"Cannot run experiment: {exc}")

if "exp" not in st.session_state:
    st.subheader("Experimental design")
    st.markdown("""
**A: Strict baseline.** Parallel fifths, parallel octaves, and leaps above the threshold are forbidden. The common structural constraints also enforce ranges, no crossing, spacing, and chord completeness.

**B: Fixed-soft baseline.** The same three transition rules become constant numerical penalties.

**C: Non-neural adaptive control.** Soft penalties follow a predetermined middle-of-phrase relaxation curve.

**D: Neural adaptive control.** Penalties scale with normalized next-note predictive entropy from a small Transformer. Low entropy reduces penalties, but never below a minimum. **This is a hypothesis, not evidence of artistic intent.**

All four share identical chord progression, candidate lattice, note preference objective, and algorithm. Each outcome is an optimal Viterbi path **within the truncated candidate lattice**. This is not a proof of fixed-parameter tractability.
""")
    st.stop()

results, tempo, model_choice, progression, config = st.session_state["exp"]
valid={k:r for k,r in results.items() if isinstance(r,Result)}
for k,r in results.items():
    if isinstance(r,str):
        st.warning(f"{METHODS[k]}: {r}")
if not valid:
    st.stop()

st.success(f"Compared {len(valid)} systems on: {progression} | {config.meter}/4 | {config.events_per_bar} event(s) per bar | predictor: {model_choice}")
if model_choice=="Bundled synthetic Transformer":
    try:
        metadata=json.loads(DEFAULT_MODEL.with_suffix('.json').read_text())
        st.caption(f"Bundled model provenance: {metadata['source']}, {metadata['examples']} synthetic transitions, {metadata['epochs']} epochs. Model is for functional demonstration, not experimental validation of human musical expressiveness.")
    except Exception:
        pass
elif model_choice=="Heuristic (debug only)":
    st.warning("System D is **not a neural-uncertainty experiment** in heuristic mode. Use this only for software debugging.")

summary=pd.DataFrame([summarize(r) for r in valid.values()])
tab1,tab2,tab3,tab4 = st.tabs(["Metrics and comparisons","Pitch trajectories","Inspect and listen","Research notes"])
with tab1:
    st.subheader("A/B/C/D metrics")
    n_unique = len({tuple(v.pitches for v in r.path) for r in valid.values()})
    if n_unique == 1:
        st.warning("All systems selected the same sequence with these settings. That is a legitimate outcome, not a plotting error. Try the dramatic-climax objective, a lower leap threshold, or weaker soft penalties to study trade-offs.")
    else:
        st.info(f"{n_unique} distinct complete sequences selected across {len(valid)} systems. Differences alone do not establish musical improvement.")
    st.dataframe(summary, hide_index=True, use_container_width=True)
    left,right=st.columns(2)
    with left:
        fig=go.Figure()
        labels=["Parallel fifths","Parallel octaves","Large leaps"]
        for name,r in valid.items():
            m=summarize(r)
            fig.add_bar(name=name,x=labels,y=[m['Parallel fifths (count)'],m['Parallel octaves (count)'],m['Large leaps (count)']])
        fig.update_layout(barmode='group',title='Constraint violations by system',xaxis_title='Constraint type',yaxis_title='Violations (count)',legend_title='System')
        st.plotly_chart(fig,use_container_width=True)
    with right:
        fig=go.Figure()
        for name,r in valid.items():
            m=summarize(r)
            fig.add_bar(name=name,x=[name],y=[m['Mean motion (semitones/transition)']])
        fig.update_layout(title='Average voice movement',xaxis_title='System',yaxis_title='Mean movement (semitones per transition)',showlegend=False)
        st.plotly_chart(fig,use_container_width=True)
    left,right=st.columns(2)
    with left:
        fig=go.Figure()
        for name,r in valid.items():
            fig.add_bar(name=name,x=[name],y=[summarize(r)['Soprano pitch entropy (bits)']])
        fig.update_layout(title='Soprano pitch variety (not artistic quality)',xaxis_title='System',yaxis_title='Shannon pitch entropy (bits)',showlegend=False)
        st.plotly_chart(fig,use_container_width=True)
    with right:
        fig=go.Figure()
        for name,r in valid.items():
            fig.add_bar(name=name,x=[name],y=[r.runtime_s])
        fig.update_layout(title='Measured search runtime',xaxis_title='System',yaxis_title='Wall-clock runtime (seconds)',showlegend=False)
        st.plotly_chart(fig,use_container_width=True)
    if 'A' in valid:
        ref = valid['A'].path
        differences = []
        for name,r in valid.items():
            changed=sum(p!=q for a,b in zip(ref,r.path) for p,q in zip(a.pitches,b.pitches))
            deviations=sum(abs(p-q) for a,b in zip(ref,r.path) for p,q in zip(a.pitches,b.pitches))
            differences.append({'System':name,'Changed pitch cells vs A (count)':changed,
                                'Absolute MIDI difference vs A (semitones)':deviations})
        st.markdown('**Direct comparison with hard baseline A**')
        st.dataframe(pd.DataFrame(differences),hide_index=True,use_container_width=True)
    st.download_button("Download metrics CSV",summary_csv(results),"abcd_metrics.csv","text/csv")

with tab2:
    st.subheader("Four pitch trajectories on the same progression")
    fig=make_subplots(rows=2,cols=2,subplot_titles=[METHODS[m] for m in METHODS])
    for index,method in enumerate(METHODS):
        r=valid.get(method)
        if r is None: continue
        row,col=index//2+1,index%2+1
        for voice_idx,voice in enumerate(('Soprano','Alto','Tenor','Bass')):
            pitches=[v.pitches[voice_idx] for v in r.path]
            fig.add_trace(go.Scatter(x=list(range(1,len(pitches)+1)),y=pitches,mode='lines+markers',
                                     name=voice,legendgroup=voice,showlegend=(index==0),
                                     customdata=[note_name(p) for p in pitches],
                                     hovertemplate="Event %{x}<br>Pitch %{customdata} (MIDI %{y})<extra>"+voice+"</extra>"),row=row,col=col)
        fig.update_xaxes(title_text='Note-event index',row=row,col=col)
        fig.update_yaxes(title_text='Pitch (MIDI number)',range=[39,85],row=row,col=col)
    fig.update_layout(height=770,title='SATB voice movement: A vs B vs C vs D',legend_title='Voice')
    st.plotly_chart(fig,use_container_width=True)
    if 'A' in valid:
        st.subheader("Pitch differences relative to strict baseline A")
        methods = [m for m in ('B','C','D') if m in valid]
        if methods:
            baseline=valid['A']
            diff_fig=make_subplots(rows=len(methods),cols=1,
                                   subplot_titles=[f"System {m} minus System A" for m in methods],
                                   vertical_spacing=.10)
            for i,m in enumerate(methods):
                y_labels=['Soprano','Alto','Tenor','Bass']
                z=[[valid[m].path[event].pitches[v]-baseline.path[event].pitches[v]
                    for event in range(len(baseline.path))] for v in range(4)]
                diff_fig.add_trace(go.Heatmap(x=list(range(1,len(baseline.path)+1)),
                                              y=y_labels,z=z,zmid=0,colorscale='RdBu',
                                              colorbar=dict(title='Difference (semitones)') if i==0 else None,
                                              showscale=(i==0),
                                              hovertemplate='Event %{x}<br>Voice %{y}<br>Pitch difference %{z} semitones<extra></extra>'),
                                   row=i+1,col=1)
                diff_fig.update_xaxes(title_text='Note-event index',row=i+1,col=1)
                diff_fig.update_yaxes(title_text='Voice',row=i+1,col=1)
            diff_fig.update_layout(height=230*len(methods)+85,
                                   title='Changed notes: nonzero cells identify actual differences')
            st.plotly_chart(diff_fig,use_container_width=True)
    fig=make_subplots(rows=2,cols=1,subplot_titles=['Uncertainty used along the chosen path' ,'Soft-rule penalty multiplier'],vertical_spacing=.16)
    for system,r in valid.items():
        x=list(range(2,len(r.path)+1))
        fig.add_trace(go.Scatter(x=x,y=[z.uncertainty for z in r.steps],mode='lines+markers',name=system,legendgroup=system),row=1,col=1)
        fig.add_trace(go.Scatter(x=x,y=[z.multiplier for z in r.steps],mode='lines+markers',name=system,legendgroup=system,showlegend=False),row=2,col=1)
    fig.update_xaxes(title_text='Note-event index',row=1,col=1)
    fig.update_xaxes(title_text='Note-event index',row=2,col=1)
    fig.update_yaxes(title_text='Predictive entropy (normalized 0–1)',range=[0,1.05],row=1,col=1)
    fig.update_yaxes(title_text='Penalty weight multiplier (0–1)',range=[0,1.05],row=2,col=1)
    fig.update_layout(height=650,legend_title='System')
    st.plotly_chart(fig,use_container_width=True)

with tab3:
    method=st.selectbox('Select system for detailed notes and audio', list(valid),format_func=lambda x: METHODS[x])
    r=valid[method]
    st.markdown('**Sequence of note events**')
    st.dataframe(notes_dataframe(r),hide_index=True,use_container_width=True)
    st.markdown('**Transition-by-transition constraint audit**')
    st.dataframe(trace_dataframe(r),hide_index=True,use_container_width=True)
    st.caption('Pitch is quantized to MIDI semitones. Event duration is uniform; repeated notes are merged when exporting MIDI.')
    col1,col2=st.columns(2)
    with col1:
        st.download_button(f'Download {method} MIDI',midi_bytes(r,tempo),f'system_{method}.mid','audio/midi')
        st.download_button(f'Download {method} notes CSV',notes_dataframe(r).to_csv(index=False),f'system_{method}_notes.csv','text/csv')
    with col2:
        st.download_button(f'Download {method} audit CSV',trace_dataframe(r).to_csv(index=False),f'system_{method}_transitions.csv','text/csv')
    if st.checkbox('Create synthesized audio preview',value=False):
        st.audio(preview_wav(r,tempo),format='audio/wav')
        st.caption('Simple reference sine-wave synthesis; MIDI output is preferred for soundfonts/DAWs.')

with tab4:
    st.markdown('''
### Research interpretation

**Research question:** Does model-derived uncertainty improve expressive variation or listener preference when used to scale selected soft musical penalties, compared to fixed and heuristic-adaptive weights?

**Critical caveats**
- The bundled checkpoint was trained on **synthetic** melodies. It validates the pipeline but cannot establish emotional/artistic confidence.
- Output predictive entropy is **not attention entropy**. Neither constitutes direct evidence of intentional rule-breaking.
- Dynamic programming is exact only over the **bounded, preselected candidate sets**. This project makes **no FPT claim**.
- Hard musical invariants are range, voice order, maximum spacing, chord membership/completeness (and optional root-bass). Only selected transition preferences are softened.
- A single music rule violation is not automatically expressive. Larger diversity/entropy is not automatically better.
- Voices change on a regular grid. Within-bar syncopation, non-chord tones, cadences, rhythmic rests, articulation and dissonance handling are not modeled here.

### Suggested evaluation protocol

1. Use a licensed, real symbolic corpus to train a neural next-note model. Split train/validation/test **by composition** and prevent near-duplicate leakage.
2. Keep model, candidate lattice, random seeds, harmonic context, and preferences fixed across A/B/C/D.
3. Run across **many** progressions and controlled hyperparameter settings. Report mean, dispersion, confidence intervals, failure rate and runtime.
4. Include ablations: fixed vs shuffled uncertainty, output entropy vs top-two probability margin, inverted gating and constant penalty multiplier.
5. Conduct a blinded listening study rating coherence, perceived intentional tension, expressive interest and preference; do not use entropy as a proxy for those.
6. Record whether relaxed constraints produce useful musical choices rather than merely extra forbidden parallels.

### Algorithmic methodology

Each time step is a synchronized SATB pitch assignment. A lattice of feasible static assignments is generated. At each transition, we compute parallel perfect intervals, leaps and movement. A discards forbidden transitions. B assigns static penalty weights. C uses a predetermined middle-of-phrase schedule. D uses neural output entropy conditioned on the previous soprano and harmonic roots. All four use the **same additive base objective** (neural next-note likelihood + motion/repetition + selectable deterministic melodic target). The optional single-event climax is a controlled stress test, not a learned expressive preference. A Viterbi-style DP finds the least-cost reachable path.

### Reproducibility

Save all CSV outputs alongside model checkpoint hash, dataset provenance, random seed, parameters and package versions. The code is a research scaffold, not a publication result.
''')
    st.code('python train_model.py --midi-dir ./licensed_midis --epochs 10 --output models/my_music.pt', language='bash')
