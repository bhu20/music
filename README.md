# Simple Bach Chorale Corpus Pipeline

This project now uses one simple data source only: the built-in Bach chorale corpus in `music21`.

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run The Pipeline

```powershell
python scripts/build_chorale_instances.py --start 1 --end 26 --numbering riemenschneider --target-voices 4 --include-roman --output-dir data/chorales_easy
```

## What This Produces

- `data/chorales_easy/instances/*.json` (one file per chorale)
- `data/chorales_easy/chorale_instances.jsonl` (all instances in one file)
- `data/chorales_easy/summary.json` (run summary)

## What Is Inside Each Instance

- Chord roots and chord durations from `chordify()`
- Starting pitches (voice-ordered)
- Constraints (`max_melodic_step`, `forbidden_parallels`, `max_k`, `max_k1`, `max_k2`, `max_k3`)
- Metadata with:
  - observed tangling (`k`, `k1`, `k2`, `k3`)
  - voice-leading interval stats
  - counterpoint checks (forbidden parallels and bass contrary motion)
  - Roman numerals when available (for early chorales)

## Optional: Strict Filtering

If you only want chorales that satisfy no forbidden parallels and enough contrary bass motion:

```powershell
python scripts/build_chorale_instances.py --start 1 --end 26 --numbering riemenschneider --target-voices 4 --include-roman --output-dir data/chorales_easy --require-no-forbidden-parallels --require-bass-contrary --bass-contrary-threshold 0.5
```

## Data Contract

Schema file:

- `schemas/problem_instance.schema.json`
