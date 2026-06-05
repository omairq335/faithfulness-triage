#!/usr/bin/env python3
"""CLI tool for triaging annotation data for post-hoc rationalization risk."""

import argparse
import json
import sys
from pathlib import Path

from scorer import score_sample


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Score annotation samples for post-hoc rationalization risk and "
            "output a ranked triage report."
        )
    )
    parser.add_argument(
        "input",
        help="Path to a JSON file containing an array of annotation samples.",
    )
    parser.add_argument(
        "--output",
        default="triage_report.json",
        help="Output path for the ranked triage report (default: triage_report.json).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        metavar="N",
        help="Only include the top N highest-risk samples in the report.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        metavar="T",
        help="Only include samples with risk_score >= T in the report.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    with open(input_path, encoding="utf-8") as f:
        try:
            samples = json.load(f)
        except json.JSONDecodeError as exc:
            print(f"error: invalid JSON in {input_path}: {exc}", file=sys.stderr)
            sys.exit(1)

    if not isinstance(samples, list):
        print("error: input file must contain a JSON array of samples", file=sys.stderr)
        sys.exit(1)

    total = len(samples)
    print(f"Scoring {total} sample(s)…", file=sys.stderr)

    results = []
    for i, sample in enumerate(samples, 1):
        preview = sample.get("prompt", "")[:70].replace("\n", " ")
        print(f"  [{i}/{total}] {preview}", file=sys.stderr)
        try:
            result = score_sample(sample)
        except Exception as exc:  # noqa: BLE001
            print(f"    warning: scoring failed — {exc}", file=sys.stderr)
            continue
        results.append(result)

    results.sort(key=lambda r: r["risk_score"], reverse=True)

    if args.threshold is not None:
        results = [r for r in results if r["risk_score"] >= args.threshold]

    if args.top is not None:
        results = results[: args.top]

    output_path = Path(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    high = sum(1 for r in results if r["risk_score"] >= 0.7)
    medium = sum(1 for r in results if 0.4 <= r["risk_score"] < 0.7)
    low = sum(1 for r in results if r["risk_score"] < 0.4)

    print(f"\nReport written to: {output_path}", file=sys.stderr)
    print(f"  High risk   (≥ 0.7): {high}", file=sys.stderr)
    print(f"  Medium risk (0.4–0.7): {medium}", file=sys.stderr)
    print(f"  Low risk    (< 0.4):  {low}", file=sys.stderr)


if __name__ == "__main__":
    main()
