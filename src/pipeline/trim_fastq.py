from __future__ import annotations

import argparse
import json
import logging
import re
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from src.utils.io import ensure_dirs

LOGGER = logging.getLogger(__name__)

try:  # Optional dependency for nicer progress bars
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None


class _DummyProgress:
    def update(self, n: int = 1) -> None:
        return

    def close(self) -> None:
        return


class _SimpleProgress:
    def __init__(self, total: int, label: str) -> None:
        self.total = total
        self.count = 0
        self.label = label
        self._print()

    def update(self, n: int = 1) -> None:
        self.count += n
        self._print()

    def _print(self) -> None:
        sys.stdout.write(f"\r{self.label}: {self.count}/{self.total}")
        sys.stdout.flush()

    def close(self) -> None:
        sys.stdout.write("\n")
        sys.stdout.flush()


class _TqdmProgress:
    def __init__(self, total: int, label: str) -> None:
        self._bar = tqdm(total=total, desc=label, unit="file")

    def update(self, n: int = 1) -> None:
        self._bar.update(n)

    def close(self) -> None:
        self._bar.close()


@dataclass(frozen=True)
class TrimJob:
    inputs: Tuple[Path, ...]
    basename: str
    paired: bool


@dataclass
class TrimResult:
    inputs: Tuple[Path, ...]
    outputs: Tuple[Path, ...]
    status: str
    returncode: Optional[int]
    duration_sec: float
    stdout: Optional[str]
    stderr: Optional[str]
    basename: str
    filename: Optional[str]
    sample_id: Optional[str]
    subject_id: Optional[str]
    sample_type: Optional[str]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trim FASTQ files with an external tool (e.g., fastp)")
    parser.add_argument(
        "--fastq-root",
        type=Path,
        default=Path("data/decoded"),
        help="Root directory containing decoded FASTQ files",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/trimmed"),
        help="Directory where trimmed FASTQ files will be written",
    )
    parser.add_argument(
        "--tool-bin",
        type=str,
        default="fastp",
        help="Trimming executable",
    )
    parser.add_argument(
        "--command-template",
        type=str,
        default="{tool} -i {input} -o {output}",
        help=(
            "Command template for trimming. Available placeholders include: {tool}, {input}, {output}, {input_r1}, {input_r2}, "
            "{output_r1}, {output_r2}, {basename}, {output_dir}, {output_root}."
        ),
    )
    parser.add_argument(
        "--train-index",
        type=Path,
        default=Path("data/Train.csv"),
        help="Optional CSV to map filenames to SampleID/Subject metadata",
    )
    parser.add_argument(
        "--input-glob",
        type=str,
        default="*.fastq*",
        help="Glob pattern (relative to fastq-root) for locating FASTQ files",
    )
    parser.add_argument(
        "--pair-regex",
        type=str,
        default=r"(?P<basename>.+?)[._-](?P<read>R?[12])(?:\.fastq|\.fq)(?:\.gz)?$",
        help=(
            "Regex used to detect paired-end FASTQ files. Must define named groups 'basename' and 'read' (values 1/2)."
        ),
    )
    parser.add_argument(
        "--require-pairs",
        action="store_true",
        help="Raise an error if an R1 file is missing its R2 mate (after matching pair-regex)",
    )
    parser.add_argument(
        "--output-name-template",
        type=str,
        default="{basename}_trimmed.fastq.gz",
        help="Template for single-end trimmed output filenames",
    )
    parser.add_argument(
        "--output-name-template-r1",
        type=str,
        default="{basename}_R1_trimmed.fastq.gz",
        help="Template for paired-end R1 trimmed output filenames",
    )
    parser.add_argument(
        "--output-name-template-r2",
        type=str,
        default="{basename}_R2_trimmed.fastq.gz",
        help="Template for paired-end R2 trimmed output filenames",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of concurrent trimming processes",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a job when all target outputs already exist",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional JSONL manifest capturing trimming results",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress output",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity",
    )
    return parser.parse_args(argv)


def discover_fastq_files(root: Path, pattern: str) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"FASTQ root {root} does not exist")
    files = sorted(p for p in root.rglob(pattern) if p.is_file())
    if not files:
        LOGGER.warning("No FASTQ files matched pattern '%s' under %s", pattern, root)
    return files


def build_jobs(
    fastq_files: Iterable[Path],
    pair_regex: str,
    require_pairs: bool,
) -> List[TrimJob]:
    regex = re.compile(pair_regex)
    buckets: Dict[str, Dict[str, Path]] = {}
    singles: List[Path] = []

    for path in fastq_files:
        match = regex.search(path.name)
        if match:
            basename = match.group("basename")
            read_token = match.group("read").upper().replace("READ", "")
            read = "1" if read_token in {"1", "R1"} else "2" if read_token in {"2", "R2"} else None
            if read is None:
                singles.append(path)
                continue
            buckets.setdefault(basename, {})[read] = path
        else:
            singles.append(path)

    jobs: List[TrimJob] = []

    for basename, reads in buckets.items():
        r1 = reads.get("1")
        r2 = reads.get("2")
        if r1 and r2:
            jobs.append(TrimJob(inputs=(r1, r2), basename=basename, paired=True))
        elif require_pairs:
            missing = "R2" if r1 else "R1"
            raise ValueError(f"Missing mate {missing} for basename '{basename}'")
        else:
            remaining = r1 or r2
            if remaining:
                jobs.append(TrimJob(inputs=(remaining,), basename=basename, paired=False))

    for path in singles:
        basename = path.stem
        jobs.append(TrimJob(inputs=(path,), basename=basename, paired=False))

    return sorted(jobs, key=lambda job: (job.basename, len(job.inputs)))


def load_train_metadata(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        LOGGER.warning("Train index %s not found; manifest will omit metadata", path)
        return {}
    df = pd.read_csv(path)
    required = {"filename", "SampleID", "SubjectID", "SampleType"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Train index {path} missing required columns: {missing}")
    return {
        row["filename"]: {
            "SampleID": row.get("SampleID"),
            "SubjectID": row.get("SubjectID"),
            "SampleType": row.get("SampleType"),
        }
        for _, row in df.iterrows()
    }


def metadata_for_basename(basename: str, metadata: Dict[str, Dict[str, str]]) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
    key = f"{basename}.mgb"
    meta = metadata.get(key)
    if meta:
        return key, meta.get("SampleID"), meta.get("SubjectID"), meta.get("SampleType")
    return key, None, None, None


def compute_outputs(
    job: TrimJob,
    output_root: Path,
    single_template: str,
    r1_template: str,
    r2_template: str,
) -> Tuple[Path, Tuple[Path, ...]]:
    out_dir = output_root / job.basename
    if job.paired:
        out_r1 = out_dir / r1_template.format(basename=job.basename, stem=job.basename)
        out_r2 = out_dir / r2_template.format(basename=job.basename, stem=job.basename)
        return out_dir, (out_r1, out_r2)
    out_single = out_dir / single_template.format(basename=job.basename, stem=job.basename)
    return out_dir, (out_single,)


def build_command(template: str, **context: str) -> List[str]:
    cmd_str = template.format(**context)
    return shlex.split(cmd_str)


def run_trim_job(
    job: TrimJob,
    output_root: Path,
    tool: str,
    template: str,
    skip_existing: bool,
    single_template: str,
    r1_template: str,
    r2_template: str,
) -> TrimResult:
    output_dir, outputs = compute_outputs(job, output_root, single_template, r1_template, r2_template)
    ensure_dirs(output_dir)

    if skip_existing and all(out.exists() for out in outputs):
        LOGGER.info("Skipping %s (trimmed outputs exist)", job.basename)
        return TrimResult(
            inputs=job.inputs,
            outputs=outputs,
            status="skipped",
            returncode=None,
            duration_sec=0.0,
            stdout=None,
            stderr=None,
            basename=job.basename,
            filename=None,
            sample_id=None,
            subject_id=None,
            sample_type=None,
        )

    context: Dict[str, str] = {
        "tool": tool,
        "basename": job.basename,
        "output_dir": str(output_dir),
        "output_root": str(output_root),
    }

    if job.paired:
        context.update(
            input_r1=str(job.inputs[0]),
            input_r2=str(job.inputs[1]),
            output_r1=str(outputs[0]),
            output_r2=str(outputs[1]),
            input=str(job.inputs[0]),
            output=str(outputs[0]),
        )
    else:
        context.update(
            input=str(job.inputs[0]),
            output=str(outputs[0]),
        )

    cmd = build_command(template, **context)
    LOGGER.debug("Running command: %s", " ".join(cmd))

    start = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        duration = time.perf_counter() - start
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Trimming executable not found: {cmd[0]}") from exc

    status = "success" if proc.returncode == 0 else "failed"
    if status == "failed":
        LOGGER.error("Trimming %s failed with return code %s", job.basename, proc.returncode)
        LOGGER.debug("stderr: %s", proc.stderr)

    return TrimResult(
        inputs=job.inputs,
        outputs=outputs,
        status=status,
        returncode=proc.returncode,
        duration_sec=duration,
        stdout=proc.stdout.strip() or None,
        stderr=proc.stderr.strip() or None,
        basename=job.basename,
        filename=None,
        sample_id=None,
        subject_id=None,
        sample_type=None,
    )


def enrich_with_metadata(results: Iterable[TrimResult], metadata: Dict[str, Dict[str, str]]) -> List[TrimResult]:
    enriched: List[TrimResult] = []
    for res in results:
        filename, sample_id, subject_id, sample_type = metadata_for_basename(res.basename, metadata)
        enriched.append(
            TrimResult(
                inputs=res.inputs,
                outputs=res.outputs,
                status=res.status,
                returncode=res.returncode,
                duration_sec=res.duration_sec,
                stdout=res.stdout,
                stderr=res.stderr,
                basename=res.basename,
                filename=filename,
                sample_id=sample_id,
                subject_id=subject_id,
                sample_type=sample_type,
            )
        )
    return enriched


def write_manifest(path: Path, results: Iterable[TrimResult]) -> None:
    ensure_dirs(path.parent)
    with path.open("w", encoding="utf-8") as fh:
        for res in results:
            record = {
                "inputs": [str(p) for p in res.inputs],
                "outputs": [str(p) for p in res.outputs],
                "status": res.status,
                "returncode": res.returncode,
                "duration_sec": round(res.duration_sec, 3),
                "stdout": res.stdout,
                "stderr": res.stderr,
                "basename": res.basename,
                "filename": res.filename,
                "SampleID": res.sample_id,
                "SubjectID": res.subject_id,
                "SampleType": res.sample_type,
            }
            fh.write(json.dumps(record) + "\n")


def run_trims(
    jobs: Iterable[TrimJob],
    output_root: Path,
    tool: str,
    template: str,
    skip_existing: bool,
    single_template: str,
    r1_template: str,
    r2_template: str,
    workers: int,
    show_progress: bool,
) -> List[TrimResult]:
    job_list = list(jobs)
    total = len(job_list)
    if total == 0:
        return []

    if show_progress:
        if tqdm is not None:
            progress = _TqdmProgress(total, "Trimming")
        else:
            progress = _SimpleProgress(total, "Trimming")
    else:
        progress = _DummyProgress()

    results: List[TrimResult] = []

    if workers <= 1:
        for job in job_list:
            results.append(
                run_trim_job(
                    job,
                    output_root,
                    tool,
                    template,
                    skip_existing,
                    single_template,
                    r1_template,
                    r2_template,
                )
            )
            progress.update()
        progress.close()
        return results

    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {
            pool.submit(
                run_trim_job,
                job,
                output_root,
                tool,
                template,
                skip_existing,
                single_template,
                r1_template,
                r2_template,
            ): job
            for job in job_list
        }
        for future in as_completed(future_map):
            job = future_map[future]
            try:
                results.append(future.result())
            except Exception as exc:  # pragma: no cover - defensive logging
                LOGGER.exception("Trimming %s raised an error", job.basename)
                _, outputs = compute_outputs(job, output_root, single_template, r1_template, r2_template)
                results.append(
                    TrimResult(
                        inputs=job.inputs,
                        outputs=outputs,
                        status="error",
                        returncode=None,
                        duration_sec=0.0,
                        stdout=None,
                        stderr=str(exc),
                        basename=job.basename,
                        filename=None,
                        sample_id=None,
                        subject_id=None,
                        sample_type=None,
                    )
                )
            finally:
                progress.update()
    progress.close()
    return results


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="[%(levelname)s] %(message)s")

    fastq_files = discover_fastq_files(args.fastq_root, args.input_glob)
    if not fastq_files:
        return 0

    jobs = build_jobs(fastq_files, pair_regex=args.pair_regex, require_pairs=args.require_pairs)
    ensure_dirs(args.output_root)

    metadata = load_train_metadata(args.train_index) if args.train_index else {}

    results = run_trims(
        jobs=jobs,
        output_root=args.output_root,
        tool=args.tool_bin,
        template=args.command_template,
        skip_existing=args.skip_existing,
        single_template=args.output_name_template,
        r1_template=args.output_name_template_r1,
        r2_template=args.output_name_template_r2,
        workers=args.workers,
        show_progress=not args.no_progress,
    )

    results = enrich_with_metadata(results, metadata)

    successes = sum(r.status == "success" for r in results)
    failures = sum(r.status == "failed" for r in results)
    skips = sum(r.status == "skipped" for r in results)
    errors = sum(r.status == "error" for r in results)
    LOGGER.info(
        "Trimmed %s jobs (%s success, %s failed, %s skipped, %s errors)",
        len(results),
        successes,
        failures,
        skips,
        errors,
    )

    if args.manifest:
        write_manifest(args.manifest, results)
        LOGGER.info("Wrote manifest to %s", args.manifest)

    return 0 if failures == 0 and errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
