# Music Constraint Research Lab (A/B/C/D)

A reproducible **research prototype** that compares four constraint policies for symbolic SATB music through a Streamlit dashboard.

**A** strict hard constraints; **B** fixed soft penalties; **C** hand-scheduled adaptive penalties; **D** Transformer predictive-entropy adaptive penalties. Every system receives the **same** candidate voicings, next-note preference model, and baseline scoring objective. The program uses a Viterbi-style dynamic program to select a sequence.

## Run

Requires Python 3.10+ (tested with Python 3.11/3.12 and PyTorch CPU).

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Use the bundled **synthetically trained** lightweight Transformer to run all four methods without downloading music data. A heuristic debugging mode is also available but **is not an actual neural D baseline**.

In Streamlit choose a progression, meter, event grid, voicing cap and penalty coefficients. Click **Run systems A, B, C, D**. Review plots (all x/y axes explicitly labeled), per-note/transition tables, MIDI, CSV, and reference-synthesized audio previews.

## What it implements

- Static hard constraints shared by A–D: SATB pitch ranges, strictly ordered voices, upper voice gaps <=12 semitones (tunable), bass–tenor gap <=19 (tunable), chord membership and complete chord pitches, optional root in bass.
- Transition features: parallel perfect fifth/octave *approximated in MIDI semitones*, oversized leaps, total movement and repeated pitches.
- A forbids fifths, octaves and leaps over the configured limit. B penalizes them with fixed weights. C changes weights with a deterministic triangle-shaped middle-of-phrase relaxation. D modulates soft weights by entropy `factor = min_factor + (1 - min_factor) * normalized_entropy ** gamma`.
- All methods share a next-note probability objective, pitch movement and repeated-note regularizers, and a deterministic melodic-contour reference to ensure fair controlled comparisons.
- MIDI and a basic non-performance-quality WAV preview; inspect individual SATB paths and transition audit CSVs.
- Uniform note events per bar, with possible sustained notes when the same pitch is repeated. 2/4, 3/4 and 4/4 supported, but **the prototype does not create expressive rhythm, free notes outside harmony, rests, suspensions or melodic ornaments**.

## Important research limits

- Bundled `models/demo_synthetic.pt` was trained on **synthetic generated transitions**, not historical scores or emotionally annotated music. It is a software demo, **not proof** of perceived expressiveness or reliable artistic confidence. See `models/demo_synthetic.json`.
- Neural signal is **predictive output entropy**, not attention entropy. Attention does not justify intent. Entropy gating is a **hypothesis to test**.
- The DP is exact **only within the candidate voicing lattice selected in advance**. The generated candidate set is capped, so it is not a formal globally optimal or fixed-parameter tractable algorithm. It does **not** establish time complexity `f(k)poly(n)`.
- Acoustic validation, interval spelling, doubled leading-tone conventions, cadential voice leading, and stylistic exceptions are *not* fully represented. Fifths/octaves are detected mod 12, which is not a complete traditional-theory model.
- Meters only control bar length/regular note-event grid; there is no rhythm-generation research contribution yet.
- Mood ("sad") is **not modeled** and should not be claimed as an achievement.

## Train a local next-note model (optional)

```bash
# Rebuild the demo model:
python train_model.py --demo --epochs 5

# Train using your own licensed MIDI corpus:
python train_model.py --midi-dir ./licensed_midis --epochs 10 --output models/my_music.pt
```

In the Streamlit sidebar choose **My trained Transformer** and enter its local checkpoint path. The `midi_data()` extractor is deliberately simple: it guesses melody by mean instrument pitch and uses the lowest simultaneous note as a harmonic-root **proxy**. This is *not* a reliable harmonic annotation pipeline, and papers should use manually validated real datasets with song-level splits. Do not run untrusted PyTorch checkpoints.

## Project layout

```text
app.py                  Streamlit dashboard and labeled plots
music_lab/core.py       Chord parser, candidate voicings, SATB constraints
music_lab/search.py     The four constraint policies and DP sequence search
music_lab/neural.py     Tiny Transformer and prediction entropy
music_lab/metrics.py    Numeric measures, audit tables
music_lab/export.py     MIDI, WAV and CSV export
train_model.py          Synthetic demo and optional local MIDI training
models/                 Bundled demo weights and provenance JSON
tests/                  Unit tests for music rules, equality, DP and MIDI
```

## Research design

Independent variable: **constraint control policy** A/B/C/D. All other settings held fixed per comparison. Suggested evaluation: many compositions, disjoint held-out test set, constraint-violation rate, phrase/melodic diversity, runtime/memory, blinded listener ratings, constant/shuffled/inverted-entropy controls, uncertainty calibration, significance testing. The metrics in the UI are **descriptive**, and a violation is not equivalent to intentional expressiveness.

## Commands

```bash
pytest -q
streamlit run app.py
```

License: generated research scaffold, available for your use. MIDI training data licenses are the researcher's responsibility.

## NEW: Bach-derived dataset page (8-voice JSON integration)

Open the app normally:

```bash
streamlit run app.py
```

Choose **Bach Dataset Lab** in the left navigation. The three supplied files are already located at:

```text
data/bach/chorale_riemenschneider_001.json
data/bach/chorale_riemenschneider_002.json
data/bach/chorale_riemenschneider_003.json
```

The Bach page can import additional `chorale_*.json` files through its uploader. It parses all eight starting voices and the actual list of timed note-anchor events instead of forcing 8 voices into a 4-part SATB representation. It runs A/B/C/D on exactly the same candidate lattice, supports `k` rank-displacement control, plots eight-voice trajectories, rule violations, root/anchor agreement, pitch differences and runtime with **named X/Y axes**, and exports CSV and 8-voice MIDI for each experiment.

### Headless repeatable experiment

```bash
python run_bach_demo.py --folder data/bach --output examples/bach_results
python -m pytest -q
```

Reports and generated MIDIs for the three supplied files are bundled in `examples/bach_results`.

### CRITICAL distinction: Bach-derived problems vs original Bach score data

The supplied files are **not complete 4-voice chorale scores**. Their format includes:

- `voices: 8`, `starting_pitches` (eight values): eight-voice generated/problem representation
- `chord_roots`: one named note/anchor per event, **not the original complete chord** and not a verified soprano line
- `chord_durations`: one quarter-note-duration number per event
- `constraints`: melodic step and tangling limits
- `metadata.original_voice_count: 4`: source Bach chorales have four voices
- `metadata.voice_leading_patterns`, `counterpoint_checks` and `roman_numerals`: summary/annotations, **not aligned eight-voice score targets**

We use the anchor pitch class as a **proxy conditioning objective**. The resulting eight-voice music is an experimental model output, not Bach's original music. Anchor agreement is not score reconstruction accuracy or perceived quality. A, B, C and D comparisons demonstrate different policy behavior on Bach-derived sequences, *not* generalization to real-world music. The built-in neural weights still come from synthetic data; no Bach training has been performed. The search is exact only over a fixed capped lattice, not formally fixed-parameter tractable. The JSON does not specify meter; exported MIDI preserves input durations but intentionally contains no asserted score time signature.

For actual score-level evaluation, obtain music21/MusicXML **aligned original four-voice note events** and build a separate experiment measuring generated notes against real held-out voices. For external validity, additionally evaluate non-Bach genres and listener-rated musical interest. Three chorales are not a statistically sufficient held-out benchmark.

### Technical additions

```text
music_lab/bach.py                Validated eight-voice dataset reader, fixed lattice, 4 policies and Viterbi DP
pages/1_Bach_Dataset_Lab.py    Interactive Streamlit page with labeled X and Y axes
run_bach_demo.py              Reproducible command-line batch run, CSV/MIDI/PNG exports
data/bach/                    Three original uploaded JSON samples, unmodified
tests/test_bach.py            Dataset, constraint, reproducibility, MIDI-duration and k tests
examples/bach_results/        Precomputed real-input outputs for all three instances
```
