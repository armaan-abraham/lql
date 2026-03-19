#!/usr/bin/env python3
"""
Long-running script that submits slurm jobs from a commands file.
Commands file format: commands separated by ==== on its own line.
Each segment between delimiters runs as its own job.
"""

import argparse
import getpass
import re
import subprocess
import tempfile
import time
from pathlib import Path

RETRY_WAIT_SECONDS = 60
HI_PRIORITY_PARTITION = "iris-hi"
HI_PRIORITY_MAX_JOBS = 6
DEFAULT_JOB_LOG_DIR = Path("/iris/u/armaana/jobs/content")


def parse_commands_file(filepath: Path) -> list[str]:
    """Parse commands file, splitting on ==== delimiter."""
    content = filepath.read_text()
    segments = content.split("\n====\n")
    # Filter out empty segments
    return [seg for seg in segments if seg.strip()]


def expand_seed_range(command: str) -> list[str]:
    """
    Expand seed range notation like --seed=[0-3] into multiple commands.
    Returns a list of commands with the range replaced by each seed value.
    Asserts that the notation is used at most once per command.
    """
    pattern = r"--seed=\[(\d+)-(\d+)\]"
    matches = list(re.finditer(pattern, command))

    if not matches:
        return [command]

    assert len(matches) == 1, (
        f"Seed range notation --seed=[X-Y] can only be used once per command, "
        f"found {len(matches)} occurrences"
    )

    match = matches[0]
    start = int(match.group(1))
    end = int(match.group(2))

    expanded = []
    for seed in range(start, end + 1):
        expanded_cmd = command[: match.start()] + f"--seed={seed}" + command[match.end() :]
        expanded.append(expanded_cmd)

    return expanded


def expand_all_commands(commands: list[str]) -> list[str]:
    """Expand seed ranges in all commands."""
    expanded = []
    for cmd in commands:
        expanded.extend(expand_seed_range(cmd))
    return expanded


def collapse_seed_commands(commands: list[str]) -> list[str]:
    """
    Collapse consecutive commands that differ only in --seed=N back into
    --seed=[min-max] range notation. Non-consecutive or non-seed commands
    are left as-is.
    """
    if not commands:
        return []

    seed_pattern = re.compile(r"--seed=(\d+)")

    def get_seed_and_template(cmd: str) -> tuple[int, str] | None:
        m = seed_pattern.search(cmd)
        if not m:
            return None
        seed = int(m.group(1))
        template = cmd[: m.start()] + "--seed={}" + cmd[m.end() :]
        return seed, template

    collapsed = []
    i = 0
    while i < len(commands):
        parsed = get_seed_and_template(commands[i])
        if parsed is None:
            collapsed.append(commands[i])
            i += 1
            continue

        seed_start, template = parsed
        seed_end = seed_start
        j = i + 1
        while j < len(commands):
            parsed_next = get_seed_and_template(commands[j])
            if parsed_next is None:
                break
            next_seed, next_template = parsed_next
            if next_template != template or next_seed != seed_end + 1:
                break
            seed_end = next_seed
            j += 1

        if seed_start == seed_end:
            collapsed.append(commands[i])
        else:
            collapsed.append(template.format(f"[{seed_start}-{seed_end}]"))
        i = j

    return collapsed


def count_running_jobs_in_partition(partition: str) -> int:
    """Count the number of running jobs for the current user in a partition."""
    user = getpass.getuser()
    result = subprocess.run(
        ["squeue", "-u", user, "-p", partition, "-h"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"squeue failed: {result.stderr}"
    lines = [line for line in result.stdout.strip().split("\n") if line]
    return len(lines)


def submit_job(commands: str, template: str, job_log_dir: Path) -> bool:
    """
    Attempt to submit a job with the given commands.
    Returns True if successful, False if hit QOS limit.
    Raises exception for other errors.
    """
    job_script = template.format(commands=commands)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
        f.write(job_script)
        tmp_path = f.name

    try:
        result = subprocess.run(
            ["sbatch", tmp_path],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            output = result.stdout.strip()
            print(f"Submitted job: {output}")
            # Extract job ID and save job content
            # sbatch output format: "Submitted batch job 123456"
            job_id = output.split()[-1]
            job_log_dir.mkdir(parents=True, exist_ok=True)
            (job_log_dir / f"{job_id}.txt").write_text(job_script)
            return True

        stderr = result.stderr
        if "QOSMaxSubmitJobPerUserLimit" in stderr:
            print(f"Hit QOS limit, will retry...")
            return False

        # Some other error
        raise RuntimeError(f"sbatch failed: {stderr}")

    finally:
        Path(tmp_path).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description="Submit slurm jobs from a commands file"
    )
    parser.add_argument(
        "commands_file",
        type=Path,
        help="Path to txt file with commands (delimited by ====)",
    )
    parser.add_argument(
        "--template",
        type=Path,
        help="Path to custom job template file (must contain {commands} placeholder)",
    )
    parser.add_argument(
        "--hi-template",
        type=Path,
        help="Path to high priority job template file for iris-hi partition",
    )
    parser.add_argument(
        "--job-log-dir",
        type=Path,
        default=DEFAULT_JOB_LOG_DIR,
        help=f"Directory to save submitted job scripts (default: {DEFAULT_JOB_LOG_DIR})",
    )
    parser.add_argument(
        "--hi-mode",
        choices=["cap", "all"],
        default="cap",
        help=(
            "When only --hi-template is provided: "
            "'cap' (default) only submits if fewer than {} hi jobs are running, "
            "'all' submits all jobs to hi regardless of queue size"
        ).format(HI_PRIORITY_MAX_JOBS),
    )
    args = parser.parse_args()

    if not args.commands_file.exists():
        raise FileNotFoundError(f"Commands file not found: {args.commands_file}")


    job_template = None
    if args.template:
        if not args.template.exists():
            raise FileNotFoundError(f"Template file not found: {args.template}")
        job_template = args.template.read_text()

    hi_template = None
    if args.hi_template:
        if not args.hi_template.exists():
            raise FileNotFoundError(f"High priority template not found: {args.hi_template}")
        hi_template = args.hi_template.read_text()
    
    assert job_template or hi_template, "At least one of --template or --hi-template must be provided"

    command_segments = parse_commands_file(args.commands_file)
    command_segments = expand_all_commands(command_segments)
    print(f"Found {len(command_segments)} job(s) to submit")

    idx = 0
    try:
        while idx < len(command_segments):
            commands = command_segments[idx]
            print(f"\nSubmitting job {idx + 1}/{len(command_segments)}...")

            # Try high priority partition first if enabled
            if hi_template:
                running_hi = count_running_jobs_in_partition(HI_PRIORITY_PARTITION)
                print(f"High priority partition has {running_hi} running jobs")
                hi_under_cap = running_hi < HI_PRIORITY_MAX_JOBS
                # When both templates: use hi if under cap, else fall back to regular.
                # When only hi template: depends on --hi-mode.
                if job_template:
                    use_hi = hi_under_cap
                else:
                    use_hi = True if args.hi_mode == "all" else hi_under_cap
                if use_hi:
                    print(f"Submitting to high priority partition...")
                    if submit_job(commands, hi_template, args.job_log_dir):
                        idx += 1
                        continue

            # Fall back to regular partition
            if job_template and submit_job(commands, job_template, args.job_log_dir):
                idx += 1
            else:
                print(f"Waiting {RETRY_WAIT_SECONDS}s before retry...")
                time.sleep(RETRY_WAIT_SECONDS)
    except (Exception, KeyboardInterrupt) as e:
        remaining = command_segments[idx:]
        if remaining:
            collapsed = collapse_seed_commands(remaining)
            remain_path = args.commands_file.parent / f"REMAIN-{args.commands_file.name}"
            remain_path.write_text("\n====\n".join(collapsed))
            print(f"\nSaved {len(collapsed)} remaining job(s) to {remain_path}")
        raise

    print(f"\nAll {len(command_segments)} jobs submitted successfully!")


if __name__ == "__main__":
    main()
