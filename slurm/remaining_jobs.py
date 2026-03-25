#!/usr/bin/env python3
"""
Create a new jobs file containing only the jobs that haven't been submitted yet.

Usage:
    python remaining_jobs.py jobs.txt 15

This means job 15 is the next job that was NOT submitted (as printed by
submit_jobs.py: "Submitting job 15/30..."). The output file will contain
jobs 15 through the end.
"""

import argparse
from pathlib import Path

from submit_jobs import (
    collapse_seed_commands,
    expand_all_commands,
    parse_commands_file,
)


def main():
    parser = argparse.ArgumentParser(
        description="Extract remaining unsubmitted jobs from a commands file"
    )
    parser.add_argument(
        "commands_file",
        type=Path,
        help="Path to the original commands file",
    )
    parser.add_argument(
        "next_job",
        type=int,
        help="The 1-indexed number of the next unsubmitted job (from 'Submitting job X/Y')",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output file path (default: REMAIN-<input filename>)",
    )
    args = parser.parse_args()

    commands = parse_commands_file(args.commands_file)
    commands = expand_all_commands(commands)
    total = len(commands)

    if args.next_job < 1 or args.next_job > total:
        parser.error(f"next_job must be between 1 and {total}, got {args.next_job}")

    remaining = commands[args.next_job - 1 :]
    collapsed = collapse_seed_commands(remaining)

    output_path = args.output or (
        args.commands_file.parent / f"REMAIN-{args.commands_file.name}"
    )
    output_path.write_text("\n====\n".join(collapsed))

    print(f"Wrote {len(remaining)} remaining job(s) ({len(collapsed)} collapsed) to {output_path}")


if __name__ == "__main__":
    main()
