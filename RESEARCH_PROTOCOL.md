# Protocol for a defensible research study

## 1. Aim

Test whether Transformer predictive output entropy is a useful context signal for relaxing *selected* symbolic-music transition preferences within an otherwise fixed formal search.

- **Do not claim:** confidence = artistic intention; any rule violation = tension; this prototype is globally optimal; bounded candidate count proves FPT.
- **Research hypothesis:** dynamic neural-uncertainty weights yield greater listener-preferred interest or a better controlled constraint/diversity tradeoff than fixed and non-neural adaptive weights.
- **Null hypothesis:** uncertainty-adaptive weights do not outperform constant or model-independent adaptive weights after controlling for the base model and candidate lattice.

## 2. Independent variable

| Method | Transition rule logic |
|---|---|
| A | Hard rejection for parallel fifths, parallel octaves, leaps above threshold |
| B | Constant positive cost for each violation |
| C | Same costs multiplied by a fixed, deterministic middle-of-phrase schedule |
| D | Same costs multiplied by `lambda_min + (1-lambda_min) * H_t**gamma` where H_t is normalized predicted next-soprano pitch entropy |

All four share harmonic inputs, static rules, chord voices, candidate lists, neural note likelihood and global preference targets.

## 3. Controlled variables

Fixed train/validation/test split **by composition**, encoder checkpoint, note-event grid, voice register, candidate lattice seed and cap, progression, tempo, style metadata and search objective coefficients. Freeze the best hyperparameters using validation data, not test results.

## 4. Measurements

**Mechanism:** rule violations, pitch and melodic interval distributions, variety/distance across samples, proportion of transitions relaxed, relationships between entropy and adjustment factor.

**Listener-centered:** blinded pairwise ratings of coherence, intentional expressive tension, melodic interest, naturalness and preference. A higher note entropy or more fifth violations is *not* itself musical improvement. Use multiple listeners, randomize conditions/order, and report effect sizes and intervals.

**Engineering:** wall clock time, number of candidate states/edges, memory usage, failure/no-solution rate. Compare at multiple cap sizes and chord lengths. No FPT theorem is established by those charts.

## 5. Ablations

1. Keep all A/B/C/D settings unchanged but shuffle entropy across time.
2. Replace neural entropy with constant multiplier matched to D's mean.
3. Invert the entropy mapping, so uncertain positions relax more.
4. Compare entropy to top-two logit margin and temperature-calibrated probabilities.
5. Remove one transition penalty at a time (fifth/octave/leap).
6. Test on held-out time signatures and keys without retuning.
7. Compare fine-grained per-voice entropy to the current soprano-only signal.

## 6. Literature and musical limits

Rule validity in this scaffold is limited to mod-12 MIDI pitch arithmetic and a simplified SATB convention. Real counterpoint requires spelling-sensitive intervals, dissonance/resolution and context-aware rules. The demo model was synthetically trained. A more meaningful research test will need independently validated MusicXML/MIDI score annotations and musical expert review.

## 7. Immediate milestones

1. **V1:** Static + transition rule tests and shared lattice, done.
2. **V2:** Four comparable systems, charts, MIDI and audit exports, done.
3. **V3:** Real symbolic corpus, train/validation/test split and improved harmonic conditioning.
4. **V4:** Uncertainty calibration plus controlled ablations.
5. **V5:** Human listening study; statistically report results and limitations.

## 8. Bach-derived JSON input pilot (added)

Three real files (`chorale_riemenschneider_001`–`003`) were integrated as **Bach-derived eight-voice problem instances**, not verified score transcriptions. The system uses the `starting_pitches`, pitch-class anchors derived from `chord_roots`, quarter-note durations, `max_melodic_step` and bounded crossing rank displacement `k`. The supplied original work had **four** parts, but these JSON problems have **eight** voices, and contain no aligned eight-voice target pitches. The input root notes must not be misrepresented as complete chords. Therefore conclusions concern program behavior on this input format only.

### Measurable outputs

- A/B/C/D rule-violation audits at each event
- mean movement, eight-voice pitch trajectories, average anchor agreement (explicitly a proxy), changes relative to A
- elapsed runtime, exported MIDI duration compared against the input JSON
- varying `k` and candidate cap; fixed-lattice DP costs, **not an FPT proof**

### Known confounds

- D model trained on **synthetic** examples, not on these Bach chorales.
- Candidate lattice is constructed near the eight seed pitches rather than from full harmonic events; tests are intentionally not musical-style evaluation.
- The constraint checker detects parallel perfect intervals by MIDI chromatic class; full diatonic spelling and asynchronous note transitions are outside the scope.
- A smooth anchor-conditioned melody can be musically poor, and more leaps do not automatically create good tension.
- Supplied metadata counterpoint ratios refer to source-derived observations and are not aligned reference outputs for the new eight voices.

### Valid next stage

Acquire full original **SATB voice tracks with note onsets, offsets, MIDI pitches and ties** from authoritative, rights-cleared MusicXML or MIDI. Train/validate/test across **disjoint works**, avoid contamination, and quantify held-out note recovery, transition errors, calibrated entropy, listener preference and cross-genre transfer. Treat results on three provided files as engineering integration smoke tests rather than statistically validated music-AI research.
