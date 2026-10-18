"""Streamlit page for the three supplied eight-voice Bach-derived instances."""
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from music_lab.bach import (read_problem, BachConfig, run_all, bach_metrics, bach_notes,
                            bach_midi, bach_validation)
from music_lab.neural import Predictor, HeuristicPredictor, DEFAULT_MODEL
from music_lab.search import METHODS
from music_lab.core import note_name

st.set_page_config(page_title='Bach Dataset | Music Research Lab', layout='wide',page_icon='🎹')
st.title('Bach-derived eight-voice research experiments')
st.caption('Dataset-conditioned A/B/C/D DP | actual durations | common lattice | quantitative comparisons')
st.warning('**Dataset integrity:** Each uploaded JSON has eight starting pitches and a note-anchor sequence derived from an originally four-voice Bach chorale. It does **not** contain eight ground-truth voice sequences or full Bach chord voicings. The generated MIDI is an experimental eight-voice reconstruction conditioned on anchors, **not Bach original audio** or an imitation-quality score.')

@st.cache_resource(show_spinner=False)
def predictor_for(choice,path):
    return HeuristicPredictor() if choice=='Heuristic debug (not neural)' else Predictor(path)

DATA=Path(__file__).resolve().parent.parent/'data'/'bach'
loaded=[]
errors=[]
for p in sorted(DATA.glob('chorale_*.json')):
    try:loaded.append(read_problem(p))
    except Exception as exc:errors.append(f'{p.name}: {exc}')

with st.sidebar:
    st.header('Bach test setup')
    uploads=st.file_uploader('Add more matching chorale JSON problems',type=['json'],accept_multiple_files=True)
    if uploads:
        for upload in uploads:
            try:
                problem=read_problem(upload.getvalue())
                if not any(p.sha256==problem.sha256 for p in loaded):loaded.append(problem)
            except Exception as exc:errors.append(f'{upload.name}: {exc}')
    selected=st.multiselect('Choose Bach datasets',options=[p.name for p in loaded],
                            default=[p.name for p in loaded],help='You can run 1–3 or upload more files.')
    st.markdown('**Shared search limits**')
    cap=st.slider('Candidate states per event (same in A/B/C/D)',8,72,24,4)
    seed=st.number_input('Deterministic seed',0,10000,7,1)
    k=st.slider('Maximum permitted voice rank displacement (k)',0,3,3,1,
                help='This controls candidate crossing. It does not establish FPT time complexity.')
    max_dev=st.slider('Max deviation from starting pitch (semitones)',2,10,5,1)
    st.markdown('**Soft-rule penalties**')
    p5=st.slider('Parallel fifth weight',0.0,15.0,5.0,.5)
    p8=st.slider('Parallel octave weight',0.0,15.0,6.0,.5)
    leap=st.slider('Leap exceeding JSON max_melodic_step weight',0.0,15.0,3.0,.5)
    st.markdown('**Shared melodic selection objective**')
    anchor=st.slider('Note-anchor agreement weight (proxy)',0.0,5.0,1.4,.2)
    movement=st.slider('Motion penalty',0.0,1.0,.065,.005)
    model_weight=st.slider('Predictor NLL weight',0.0,5.0,.7,.1)
    st.markdown('**Uncertainty and output**')
    signal=st.selectbox('Predictor',('Synthetic-demo Transformer','Heuristic debug (not neural)'))
    gamma=st.slider('D: predictive-entropy exponent',1.0,8.0,3.0,.5)
    tempo=st.slider('Export playback tempo (BPM, assumed)',50,150,90,5)
    start=st.button('Run all selected Bach datasets',type='primary',use_container_width=True)

for e in errors:st.error(e)
if not loaded:
    st.error('No valid JSON problems found. Put chorale_*.json in data/bach/ or upload them above.')
    st.stop()

st.subheader('Dataset integrity check')
st.dataframe(bach_validation(loaded),hide_index=True,use_container_width=True)
st.caption('Quarter-note durations are taken from the supplied files. The score time signature is not provided, so this experiment does not invent one. The model uses note anchors as conditioning labels, not verified chord progressions.')

if start:
    if not selected:
        st.error('Choose at least one dataset.')
    else:
        try:
            predictor=predictor_for(signal,str(DEFAULT_MODEL))
            config=BachConfig(cap=cap,seed=int(seed),k=k,max_displacement=max_dev,
                              pf_weight=p5,po_weight=p8,leap_weight=leap,
                              anchor_weight=anchor,movement_weight=movement,
                              note_weight=model_weight,entropy_exponent=gamma)
            with st.spinner('Running identical Bach-derived lattices through all four constraint policies...'):
                experiments={p.name:run_all(p,predictor,config) for p in loaded if p.name in selected}
            st.session_state['bach_exp']=(experiments,config,signal)
        except Exception as exc:
            st.error(f'Experiment failed: {exc}')

if 'bach_exp' not in st.session_state:
    st.info('Run the datasets to create an A/B/C/D comparison. No separate MIDI files are needed.')
    st.markdown('''### Research design

- **A**: disallow parallel fifths, parallel octaves and voice leaps exceeding the dataset threshold.
- **B**: assign fixed penalties to those violations.
- **C**: change penalties using a predetermined phrase-position schedule.
- **D**: scale penalties using next-note predictive entropy. The bundled model is **synthetically trained**; this is a software integration test, not Bach-trained uncertainty.

The supplied `max_k` limits voice-rank displacement, while the supplied `max_melodic_step` gives a transition threshold. All generated voices are allowed to move within a fixed register neighborhood, and all systems search the **same states**. This is capped-lattice DP, not a proof of FPT.

**What the experiment can establish now:** the algorithms accept the provided JSON, honor event durations and static limits, produce distinct search paths when their policies favor them, and generate auditable MIDI files. **What it cannot establish yet:** agreement with Bach's complete original note sequence, artistic superiority, or transfer to arbitrary real-life music.''')
    st.stop()

experiments,config,signal_used=st.session_state['bach_exp']
if signal_used=='Heuristic debug (not neural)':
    st.warning('D is running with heuristic output entropy in this mode; it is **not** a neural-uncertainty experiment.')
else:
    st.info('D uses the bundled **synthetically trained** Transformer, not a Bach-trained model. Treat this comparison as a functional pilot.')

rows=[bach_metrics(res) for result_set in experiments.values() for res in result_set.values()]
summary=pd.DataFrame(rows)
t1,t2,t3,t4=st.tabs(['All datasets: summary','Pitch paths: inspect','Rule audits and downloads','Methodology and limits'])
with t1:
    st.subheader('A/B/C/D measured results across supplied chorales')
    st.dataframe(summary,hide_index=True,use_container_width=True)
    types=['Parallel fifths (count)','Parallel octaves (count)','Leaps above dataset threshold (count)']
    df=summary.melt(id_vars=['Chorale','System'],value_vars=types,var_name='Constraint type',value_name='Violations (count)')
    from plotly.express import bar
    fig=bar(df,x='Constraint type',y='Violations (count)',color='System',facet_col='Chorale',barmode='group',
            title='Rule violations on each Bach-derived instance',category_orders={'System':['A','B','C','D']})
    fig.update_layout(xaxis_title='Constraint type',yaxis_title='Violations (count)',legend_title='System')
    fig.for_each_xaxis(lambda a:a.update(title_text='Constraint type'))
    fig.for_each_yaxis(lambda a:a.update(title_text='Violations (count)'))
    st.plotly_chart(fig,use_container_width=True)
    chart=bar(summary,x='System',y='Anchor match (soprano pitch-class %)',color='System',facet_col='Chorale',
              title='Pitch-class agreement with input anchors (NOT Bach transcription accuracy)')
    chart.for_each_xaxis(lambda a:a.update(title_text='System'))
    chart.for_each_yaxis(lambda a:a.update(title_text='Soprano anchor agreement (%)'))
    st.plotly_chart(chart,use_container_width=True)
    metric=st.selectbox('Choose quantitative metric to compare',[
        'Mean total movement (semitones/event)','Runtime (seconds)',
        'Leaps above dataset threshold (count)','Distinct soprano pitches (count)',
        'Max observed voice rank displacement'],key='metric_chart')
    fig=bar(summary,x='Chorale',y=metric,color='System',barmode='group',title=f'{metric} by dataset and system')
    fig.update_layout(xaxis_title='Bach-derived dataset',yaxis_title=metric,legend_title='System')
    st.plotly_chart(fig,use_container_width=True)
    st.download_button('Download all-dataset metrics CSV',summary.to_csv(index=False).encode(),
                       'bach_abcd_metrics.csv','text/csv')

with t2:
    st.subheader('Pitch movement visualization')
    selected_chorale=st.selectbox('Select dataset',list(experiments),key='pitch_dataset')
    results=experiments[selected_chorale]
    problem=results['A'].problem
    voice_idx=st.selectbox('Choose voice for direct system comparison',list(range(8)),
                            format_func=lambda x:f'Voice {x+1}',key='voice_selector')
    times=[0.0]
    for d in problem.durations[:-1]:times.append(times[-1]+d)
    fig=go.Figure()
    for method,res in results.items():
        ps=[v.pitches[voice_idx] for v in res.states]
        fig.add_trace(go.Scatter(x=times,y=ps,mode='lines+markers',name=f'System {method}',
                                 customdata=[note_name(p) for p in ps],
                                 hovertemplate='Elapsed beat %{x:.2f}<br>Pitch %{customdata} (MIDI %{y})<extra></extra>'))
    fig.update_layout(title=f'{problem.name}: Voice {voice_idx+1} compared across systems',
                      xaxis_title='Elapsed quarter-note beats (from input durations)',
                      yaxis_title='Pitch (MIDI number)',legend_title='System')
    st.plotly_chart(fig,use_container_width=True)
    st.subheader('Eight-voice pitch lines for each system')
    all_fig=make_subplots(rows=2,cols=2,subplot_titles=[METHODS[m] for m in METHODS])
    for idx,(method,res) in enumerate(results.items()):
        row,col=idx//2+1,idx%2+1
        for v in range(8):
            all_fig.add_trace(go.Scatter(x=times,y=[s.pitches[v] for s in res.states],
                                        name=f'Voice {v+1}',legendgroup=f'voice_{v}',
                                        showlegend=idx==0,mode='lines',line={'width':1.6}),row=row,col=col)
        all_fig.update_xaxes(title_text='Elapsed quarter-note beats',row=row,col=col)
        all_fig.update_yaxes(title_text='Pitch (MIDI number)',row=row,col=col)
    all_fig.update_layout(height=850,title='Eight musical voices by system',legend_title='Voice')
    st.plotly_chart(all_fig,use_container_width=True)
    ref=results['A']
    hd=go.Figure()
    alternatives=[m for m in ('B','C','D')]
    n=len(ref.states)
    for index,m in enumerate(alternatives):
        arr=[[results[m].states[t].pitches[j]-ref.states[t].pitches[j] for t in range(n)] for j in range(8)]
        if index==0:
            fig2=make_subplots(rows=3,cols=1,subplot_titles=[f'{m} minus A, MIDI pitch displacement' for m in alternatives],vertical_spacing=.085)
        fig2.add_trace(go.Heatmap(x=list(range(1,n+1)),y=[f'Voice {i+1}' for i in range(8)],z=arr,
                                  zmid=0,colorscale='RdBu',showscale=index==0,
                                  colorbar={'title':'Pitch difference (semitones)'},
                                  hovertemplate='Event %{x}<br>%{y}<br>Difference %{z} semitones<extra></extra>'),row=index+1,col=1)
        fig2.update_xaxes(title_text='Note event index',row=index+1,col=1)
        fig2.update_yaxes(title_text='Voice',row=index+1,col=1)
    fig2.update_layout(title='Difference relative to System A (zero = same pitch)',height=750)
    st.plotly_chart(fig2,use_container_width=True)

with t3:
    selected_chorale=st.selectbox('Select dataset',list(experiments),key='audit_dataset')
    selected_method=st.selectbox('Select method',['A','B','C','D'],key='audit_method')
    r=experiments[selected_chorale][selected_method]
    notes=bach_notes(r);transitions=pd.DataFrame(r.traces)
    st.markdown('**Generated notes at each timed event**')
    st.dataframe(notes,hide_index=True,use_container_width=True,height=330)
    st.markdown('**Transition audit**')
    st.dataframe(transitions,hide_index=True,use_container_width=True,height=330)
    a,b,c=st.columns(3)
    with a:st.download_button('Export 8-voice MIDI',bach_midi(r,tempo),
                              f'{selected_chorale}_{selected_method}_experimental.mid','audio/midi')
    with b:st.download_button('Export generated notes CSV',notes.to_csv(index=False).encode(),
                              f'{selected_chorale}_{selected_method}_notes.csv','text/csv')
    with c:st.download_button('Export transition audit CSV',transitions.to_csv(index=False).encode(),
                              f'{selected_chorale}_{selected_method}_audit.csv','text/csv')
    st.caption('No original time signature is present in JSON. MIDI uses the stored per-event durations and the chosen preview tempo.')
    st.markdown('**Input metadata for context, not a validation target**')
    st.json(r.problem.metadata)

with t4:
    st.markdown('''### What has actually been tested

The supplied eight-voice problem encoding is parsed without constructing fictitious four-voice Bach notes. Each event uses the supplied pitch anchor and duration. Candidate eight-note voice states are constructed around the eight provided starting pitches, and the identical fixed state lattice goes into all methods. Each run searches through all time steps using Viterbi-style DP and exports full-path results.

**A** forbids parallel perfect fifths/octaves and moves exceeding the JSON `max_melodic_step`; **B** turns them into weighted penalties; **C** schedules those penalties by phrase position; **D** weights them using a normalized next-note entropy signal. Structural register deviation and voice rank displacement are shared constraints, and their upper bound `k` can be tested.

The note-anchor matching measure is a **proxy objective**: the `chord_roots` field stores one named pitch per event, not the complete harmonic voicing or observed soprano. A higher anchor-matching percentage does not mean Bach-like style, human preference, or compositional accuracy. The original metadata (`counterpoint_checks`, `voice_leading_patterns`) describes the source-derived representation and should not be compared directly to our generated eight voices as if they were aligned note-for-note.

### Required next level for real held-out validation

1. Obtain aligned original SATB parts (note MIDI, onset and duration per voice) from licensed MusicXML/MIDI, not only these problem summaries.
2. Train on separate chorales, validate on separate chorales, and freeze all hyperparameters before touching the test chorales.
3. Run **Bach reconstruction/inpainting** on original voice sequences to measure exact-note/interval recovery, rule fidelity, and rhythmic accuracy.
4. Test non-Bach genres and conduct blinded listening studies before claiming transfer to music in general.
5. Treat the current synthesized-demo neural model as a pipeline smoke test only. Do not interpret attention or output entropy as artistic intent.

**Mathematical caveat:** We use bounded candidate lattice size, not a proven FPT solver. The controlled variable `k` is a voice rank displacement bound; a runtime chart over `k` cannot establish `f(k)poly(n)` without a separate algorithmic proof.
''')
