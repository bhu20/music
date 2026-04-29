from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mus.chorale_pipeline import iter_chorales, make_chorale_problem, write_instances
from mus.models import Constraints, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate chorale-derived benchmark instances from music21 corpus."
    )
    parser.add_argument("--start", type=int, default=1, help="First chorale index")
    parser.add_argument("--end", type=int, default=26, help="End chorale index (exclusive)")
    parser.add_argument(
        "--numbering",
        type=str,
        default="riemenschneider",
        choices=["riemenschneider", "bwv", "kalmus"],
        help="Chorale numbering system used by music21 iterator",
    )
    parser.add_argument(
        "--target-voices",
        type=int,
        default=4,
        help="Voice count in generated problem definitions",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "chorales_easy",
        help="Output directory for generated JSON data",
    )
    parser.add_argument(
        "--problem-prefix",
        type=str,
        default="chorale",
        help="Prefix used in problem_id",
    )
    parser.add_argument(
        "--max-melodic-step",
        type=int,
        default=2,
        help="Constraint: maximum melodic step in semitones",
    )
    parser.add_argument("--max-k", type=int, default=3, help="Constraint: global max k")
    parser.add_argument(
        "--max-k1",
        type=int,
        default=1,
        help="Constraint: cross-family displacement budget",
    )
    parser.add_argument(
        "--max-k2",
        type=int,
        default=3,
        help="Constraint: within-family displacement budget",
    )
    parser.add_argument(
        "--max-k3",
        type=int,
        default=7,
        help="Constraint: range deviation budget",
    )
    parser.add_argument(
        "--include-roman",
        action="store_true",
        help="Try to attach Roman numeral analyses for first Riemenschneider chorales",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of generated instances",
    )
    parser.add_argument(
        "--require-no-forbidden-parallels",
        action="store_true",
        help="Keep only instances with zero forbidden parallel motion (P5/P8 by default)",
    )
    parser.add_argument(
        "--require-bass-contrary",
        action="store_true",
        help="Keep only instances where bass is mostly in contrary motion against soprano",
    )
    parser.add_argument(
        "--bass-contrary-threshold",
        type=float,
        default=0.5,
        help="Minimum contrary_ratio_active required when --require-bass-contrary is set",
    )
    return parser

def main() -> int:
    args = build_parser().parse_args()

    constraints = Constraints(
        max_melodic_step=args.max_melodic_step,
        forbidden_parallels=["P5", "P8"],
        max_k=args.max_k,
        max_k1=args.max_k1,
        max_k2=args.max_k2,
        max_k3=args.max_k3,
    )

    instances = []
    failures: list[dict[str, str]] = []
    filtered_out: list[dict[str, str]] = []

    for ordinal, chorale in iter_chorales(
        start=args.start,
        end=args.end,
        numbering=args.numbering,
    ):
        if args.limit is not None and len(instances) >= args.limit:
            break

        problem_id = f"{args.problem_prefix}_{args.numbering}_{ordinal:03d}"

        try:
            instance = make_chorale_problem(
                chorale=chorale,
                problem_id=problem_id,
                target_voices=args.target_voices,
                base_constraints=constraints,
                include_roman=args.include_roman,
                riemenschneider_number=(
                    ordinal if args.numbering == "riemenschneider" else None
                ),
            )

            checks = instance.metadata.get("counterpoint_checks", {})
            parallel = checks.get("forbidden_parallel_motion", {})
            bass = checks.get("bass_contrary_motion", {})

            if args.require_no_forbidden_parallels and parallel.get("total", 0) > 0:
                filtered_out.append(
                    {
                        "problem_id": problem_id,
                        "reason": "forbidden_parallel_motion",
                    }
                )
                continue

            if args.require_bass_contrary:
                contrary_ratio = float(bass.get("contrary_ratio_active", 0.0))
                if contrary_ratio < args.bass_contrary_threshold:
                    filtered_out.append(
                        {
                            "problem_id": problem_id,
                            "reason": "bass_not_contrary_enough",
                        }
                    )
                    
                    continue

            instances.append(instance)
        except Exception as exc:  # pragma: no cover - data-dependent failures
            failures.append({"problem_id": problem_id, "error": str(exc)})

    if not instances:
        print("No instances were generated.")
        return 1

    jsonl_path, summary_path = write_instances(instances, args.output_dir)

    total_forbidden = sum(
        int(
            item.metadata.get("counterpoint_checks", {})
            .get("forbidden_parallel_motion", {})
            .get("total", 0)
        )
        for item in instances
    )
    contrary_ratios = [
        float(
            item.metadata.get("counterpoint_checks", {})
            .get("bass_contrary_motion", {})
            .get("contrary_ratio_active", 0.0)
        )
        for item in instances
    ]
    mean_contrary_ratio = (
        0.0 if not contrary_ratios else sum(contrary_ratios) / len(contrary_ratios)
    )

    write_json(
        summary_path,
        {
            "generated": len(instances),
            "failed": len(failures),
            "failures": failures,
            "filtered": len(filtered_out),
            "filtered_out": filtered_out,
            "jsonl": str(jsonl_path),
            "instances_dir": str(args.output_dir / "instances"),
            "numbering": args.numbering,
            "range": [args.start, args.end],
            "target_voices": args.target_voices,
            "counterpoint_summary": {
                "total_forbidden_parallel_motion": total_forbidden,
                "mean_bass_contrary_ratio_active": round(mean_contrary_ratio, 4),
            },
        },
    )

    print(f"Generated {len(instances)} chorale instances -> {jsonl_path}")
    print(
        "Counterpoint checks: "
        f"forbidden_parallels_total={total_forbidden}, "
        f"mean_bass_contrary_ratio_active={mean_contrary_ratio:.4f}"
    )
    if failures:
        print(f"Skipped {len(failures)} chorales. See summary at {summary_path}")
    if filtered_out:
        print(f"Filtered out {len(filtered_out)} chorales. See summary at {summary_path}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
