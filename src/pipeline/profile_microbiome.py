from __future__ import annotations

import argparse
import csv
import json
import logging
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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
        self._bar = tqdm(total=total, desc=label, unit="sample")

    def update(self, n: int = 1) -> None:
        self._bar.update(n)

    def close(self) -> None:
        self._bar.close()


@dataclass(frozen=True)
class ProfileJob:
    inputs: Tuple[Path, ...]
    basename: str
    paired: bool


@dataclass
class ProfileResult:
    basename: str
    status: str
    returncode: Optional[int]
    duration_sec: float
    inputs: Tuple[Path, ...]
    kraken_report: Path
    kraken_output: Path
    bracken_output: Path
    humann_dir: Path
    humann_pathabundance: Path
    logs: List[Dict[str, Optional[str]]]
    filename: Optional[str]
    sample_id: Optional[str]
    subject_id: Optional[str]
    sample_type: Optional[str]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Kraken2 → Bracken → HUMAnN profiling pipeline")
    parser.add_argument(
        "--fastq-root",
        type=Path,
        default=Path("data/trimmed"),
        help="Directory containing trimmed FASTQ files (single or paired)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/profiles"),
        help="Root directory for profiling outputs (per sample subdirectories)",
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
            "Regex used to pair reads. Must expose named groups 'basename' and 'read'; "
            "read should resolve to 1/2 (e.g., R1/R2)."
        ),
    )
    parser.add_argument(
        "--require-pairs",
        action="store_true",
        help="Raise an error if an R1 file is missing its R2 mate",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of concurrent samples to process",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip samples when HUMAnN outputs already exist",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional JSONL manifest capturing pipeline outputs per sample",
    )
    parser.add_argument(
        "--train-index",
        type=Path,
        default=Path("data/Train.csv"),
        help="CSV linking filenames to SampleID/Subject metadata for manifest enrichment",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress reporting",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity",
    )

    # Kraken2
    parser.add_argument("--kraken2-bin", type=str, default="kraken2", help="Kraken2 executable")
    parser.add_argument(
        "--kraken2-db",
        type=Path,
        required=True,
        help="Path to Kraken2 database",
    )
    parser.add_argument(
        "--kraken2-threads",
        type=int,
        default=1,
        help="Threads for Kraken2",
    )
    parser.add_argument(
        "--kraken2-extra",
        type=str,
        default="",
        help="Additional Kraken2 arguments (quoted string)",
    )

    # Bracken
    parser.add_argument("--bracken-bin", type=str, default="bracken", help="Bracken executable")
    parser.add_argument(
        "--bracken-db",
        type=Path,
        default=None,
        help="Path to Bracken database (defaults to Kraken2 DB if omitted)",
    )
    parser.add_argument(
        "--bracken-read-len",
        type=int,
        default=None,
        help="Read length for Bracken (-r)",
    )
    parser.add_argument(
        "--bracken-level",
        type=str,
        default=None,
        help="Taxonomic level for Bracken (-l)",
    )
    parser.add_argument(
        "--bracken-extra",
        type=str,
        default="",
        help="Additional Bracken arguments (quoted string)",
    )

    # HUMAnN
    parser.add_argument("--humann-bin", type=str, default="humann", help="HUMAnN executable")
    parser.add_argument(
        "--humann-threads",
        type=int,
        default=1,
        help="Threads for HUMAnN",
    )
    parser.add_argument(
        "--humann-extra",
        type=str,
        default="",
        help="Additional HUMAnN arguments (quoted string)",
    )
    parser.add_argument(
        "--humann-read",
        choices=["auto", "r1"],
        default="auto",
        help="Which read to feed HUMAnN when paired data are available",
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
) -> List[ProfileJob]:
    import re

    regex = re.compile(pair_regex)
    buckets: Dict[str, Dict[str, Path]] = {}
    singles: List[Path] = []

    for path in fastq_files:
        match = regex.search(path.name)
        if match:
            basename = match.group("basename")
            read_token = match.group("read").upper()
            if read_token in {"1", "R1", "READ1"}:
                read_key = "1"
            elif read_token in {"2", "R2", "READ2"}:
                read_key = "2"
            else:
                singles.append(path)
                continue
            buckets.setdefault(basename, {})[read_key] = path
        else:
            singles.append(path)

    jobs: List[ProfileJob] = []

    for basename, reads in buckets.items():
        r1 = reads.get("1")
        r2 = reads.get("2")
        if r1 and r2:
            jobs.append(ProfileJob(inputs=(r1, r2), basename=basename, paired=True))
        elif require_pairs:
            missing = "R2" if r1 else "R1"
            raise ValueError(f"Missing mate {missing} for basename '{basename}'")
        else:
            remaining = r1 or r2
            if remaining:
                jobs.append(ProfileJob(inputs=(remaining,), basename=basename, paired=False))

    for path in singles:
        basename = path.stem
        jobs.append(ProfileJob(inputs=(path,), basename=basename, paired=False))

    return sorted(jobs, key=lambda job: (job.basename, len(job.inputs)))


def load_train_metadata(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        LOGGER.warning("Train index %s not found; manifest will omit metadata", path)
        return {}

    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"filename", "SampleID", "SubjectID", "SampleType"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Train index {path} missing required columns: {missing}")
        return {
            row["filename"]: {
                "SampleID": row.get("SampleID"),
                "SubjectID": row.get("SubjectID"),
                "SampleType": row.get("SampleType"),
            }
            for row in reader
        }


def metadata_for_basename(basename: str, metadata: Dict[str, Dict[str, str]]) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
    key = f"{basename}.mgb"
    meta = metadata.get(key)
    if meta:
        return key, meta.get("SampleID"), meta.get("SubjectID"), meta.get("SampleType")
    return key, None, None, None


def run_command(cmd: Sequence[str]) -> Tuple[int, Optional[str], Optional[str], float]:
    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    duration = time.perf_counter() - start
    stdout = proc.stdout.strip() or None
    stderr = proc.stderr.strip() or None
    return proc.returncode, stdout, stderr, duration


def run_profile_job(
    job: ProfileJob,
    args: argparse.Namespace,
    metadata: Dict[str, Dict[str, str]],
) -> ProfileResult:
    sample_dir = args.output_root / job.basename
    kraken_dir = sample_dir / "kraken2"
    bracken_dir = sample_dir / "bracken"
    humann_dir = sample_dir / "humann"
    ensure_dirs(kraken_dir, bracken_dir, humann_dir)

    kraken_report = kraken_dir / f"{job.basename}.kraken2.report"
    kraken_output = kraken_dir / f"{job.basename}.kraken2.out"
    bracken_output = bracken_dir / f"{job.basename}.bracken.tsv"
    humann_pathabundance = humann_dir / f"{job.basename}_pathabundance.tsv"

    if args.skip_existing and humann_pathabundance.exists():
        filename, sample_id, subject_id, sample_type = metadata_for_basename(job.basename, metadata)
        return ProfileResult(
            basename=job.basename,
            status="skipped",
            returncode=None,
            duration_sec=0.0,
            inputs=job.inputs,
            kraken_report=kraken_report,
            kraken_output=kraken_output,
            bracken_output=bracken_output,
            humann_dir=humann_dir,
            humann_pathabundance=humann_pathabundance,
            logs=[],
            filename=filename,
            sample_id=sample_id,
            subject_id=subject_id,
            sample_type=sample_type,
        )

    total_timer = time.perf_counter()
    logs: List[Dict[str, Optional[str]]] = []

    # Kraken2
    cmd_kraken: List[str] = [
        args.kraken2_bin,
        "--db",
        str(args.kraken2_db),
        "--output",
        str(kraken_output),
        "--report",
        str(kraken_report),
    ]
    if args.kraken2_threads:
        cmd_kraken.extend(["--threads", str(args.kraken2_threads)])
    if args.kraken2_extra:
        cmd_kraken.extend(shlex.split(args.kraken2_extra))
    if job.paired:
        cmd_kraken.append("--paired")
        cmd_kraken.extend(str(p) for p in job.inputs)
    else:
        cmd_kraken.append(str(job.inputs[0]))

    rc, stdout, stderr, _ = run_command(cmd_kraken)
    logs.append({"step": "kraken2", "returncode": str(rc), "stdout": stdout, "stderr": stderr})
    if rc != 0:
        duration = time.perf_counter() - total_timer
        filename, sample_id, subject_id, sample_type = metadata_for_basename(job.basename, metadata)
        return ProfileResult(
            basename=job.basename,
            status="kraken2_failed",
            returncode=rc,
            duration_sec=duration,
            inputs=job.inputs,
            kraken_report=kraken_report,
            kraken_output=kraken_output,
            bracken_output=bracken_output,
            humann_dir=humann_dir,
            humann_pathabundance=humann_pathabundance,
            logs=logs,
            filename=filename,
            sample_id=sample_id,
            subject_id=subject_id,
            sample_type=sample_type,
        )

    # Bracken
    bracken_db = str(args.bracken_db or args.kraken2_db)
    cmd_bracken: List[str] = [
        args.bracken_bin,
        "-d",
        bracken_db,
        "-i",
        str(kraken_report),
        "-o",
        str(bracken_output),
    ]
    if args.bracken_read_len is not None:
        cmd_bracken.extend(["-r", str(args.bracken_read_len)])
    if args.bracken_level is not None:
        cmd_bracken.extend(["-l", args.bracken_level])
    if args.bracken_extra:
        cmd_bracken.extend(shlex.split(args.bracken_extra))

    rc, stdout, stderr, _ = run_command(cmd_bracken)
    logs.append({"step": "bracken", "returncode": str(rc), "stdout": stdout, "stderr": stderr})
    if rc != 0:
        duration = time.perf_counter() - total_timer
        filename, sample_id, subject_id, sample_type = metadata_for_basename(job.basename, metadata)
        return ProfileResult(
            basename=job.basename,
            status="bracken_failed",
            returncode=rc,
            duration_sec=duration,
            inputs=job.inputs,
            kraken_report=kraken_report,
            kraken_output=kraken_output,
            bracken_output=bracken_output,
            humann_dir=humann_dir,
            humann_pathabundance=humann_pathabundance,
            logs=logs,
            filename=filename,
            sample_id=sample_id,
            subject_id=subject_id,
            sample_type=sample_type,
        )

    # HUMAnN – default to first read unless user overrides
    humann_input = job.inputs[0]
    if job.paired and args.humann_read == "auto":
        # HUMAnN generally expects merged/forward reads; default to R1 but expose option for automation.
        humann_input = job.inputs[0]
    elif job.paired and args.humann_read == "r1":
        humann_input = job.inputs[0]

    cmd_humann: List[str] = [
        args.humann_bin,
        "--input",
        str(humann_input),
        "--output",
        str(humann_dir),
        "--threads",
        str(args.humann_threads),
        "--output-basename",
        job.basename,
    ]
    if args.humann_extra:
        cmd_humann.extend(shlex.split(args.humann_extra))

    rc, stdout, stderr, _ = run_command(cmd_humann)
    logs.append({"step": "humann", "returncode": str(rc), "stdout": stdout, "stderr": stderr})
    duration = time.perf_counter() - total_timer

    status = "success" if rc == 0 else "humann_failed"

    filename, sample_id, subject_id, sample_type = metadata_for_basename(job.basename, metadata)
    return ProfileResult(
        basename=job.basename,
        status=status,
        returncode=rc,
        duration_sec=duration,
        inputs=job.inputs,
        kraken_report=kraken_report,
        kraken_output=kraken_output,
        bracken_output=bracken_output,
        humann_dir=humann_dir,
        humann_pathabundance=humann_pathabundance,
        logs=logs,
        filename=filename,
        sample_id=sample_id,
        subject_id=subject_id,
        sample_type=sample_type,
    )


def write_manifest(path: Path, results: Iterable[ProfileResult]) -> None:
    ensure_dirs(path.parent)
    with path.open("w", encoding="utf-8") as fh:
        for res in results:
            record = {
                "basename": res.basename,
                "status": res.status,
                "returncode": res.returncode,
                "duration_sec": round(res.duration_sec, 3),
                "inputs": [str(p) for p in res.inputs],
                "kraken_report": str(res.kraken_report),
                "kraken_output": str(res.kraken_output),
                "bracken_output": str(res.bracken_output),
                "humann_dir": str(res.humann_dir),
                "humann_pathabundance": str(res.humann_pathabundance),
                "logs": res.logs,
                "filename": res.filename,
                "SampleID": res.sample_id,
                "SubjectID": res.subject_id,
                "SampleType": res.sample_type,
            }
            fh.write(json.dumps(record) + "\n")


def run_pipeline(args: argparse.Namespace) -> List[ProfileResult]:
    fastq_files = discover_fastq_files(args.fastq_root, args.input_glob)
    if not fastq_files:
        return []

    jobs = build_jobs(fastq_files, pair_regex=args.pair_regex, require_pairs=args.require_pairs)
    ensure_dirs(args.output_root)

    metadata = load_train_metadata(args.train_index) if args.train_index else {}

    progress: _DummyProgress | _SimpleProgress | _TqdmProgress
    if args.no_progress:
        progress = _DummyProgress()
    else:
        if tqdm is not None:
            progress = _TqdmProgress(len(jobs), "Profiling")
        else:
            progress = _SimpleProgress(len(jobs), "Profiling")

    results: List[ProfileResult] = []

    if args.workers <= 1:
        for job in jobs:
            results.append(run_profile_job(job, args, metadata))
            progress.update()
        progress.close()
        return results

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_map = {pool.submit(run_profile_job, job, args, metadata): job for job in jobs}
        for future in as_completed(future_map):
            try:
                results.append(future.result())
            except Exception as exc:  # pragma: no cover - defensive logging
                job = future_map[future]
                LOGGER.exception("Profiling %s raised an error", job.basename)
                sample_dir = args.output_root / job.basename
                kraken_dir = sample_dir / "kraken2"
                bracken_dir = sample_dir / "bracken"
                humann_dir = sample_dir / "humann"
                kraken_report = kraken_dir / f"{job.basename}.kraken2.report"
                kraken_output = kraken_dir / f"{job.basename}.kraken2.out"
                bracken_output = bracken_dir / f"{job.basename}.bracken.tsv"
                humann_pathabundance = humann_dir / f"{job.basename}_pathabundance.tsv"
                filename, sample_id, subject_id, sample_type = metadata_for_basename(job.basename, metadata)
                results.append(
                    ProfileResult(
                        basename=job.basename,
                        status="error",
                        returncode=None,
                        duration_sec=0.0,
                        inputs=job.inputs,
                        kraken_report=kraken_report,
                        kraken_output=kraken_output,
                        bracken_output=bracken_output,
                        humann_dir=humann_dir,
                        humann_pathabundance=humann_pathabundance,
                        logs=[{"step": "pipeline", "stderr": str(exc), "stdout": None, "returncode": None}],
                        filename=filename,
                        sample_id=sample_id,
                        subject_id=subject_id,
                        sample_type=sample_type,
                    )
                )
            finally:
                progress.update()
    progress.close()
    return results


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="[%(levelname)s] %(message)s")

    results = run_pipeline(args)
    if not results:
        LOGGER.info("No FASTQ files processed")
        return 0

    successes = sum(r.status == "success" for r in results)
    failures = sum(r.status not in {"success", "skipped"} for r in results)
    skips = sum(r.status == "skipped" for r in results)
    LOGGER.info(
        "Completed profiling for %s samples (%s success, %s failed, %s skipped)",
        len(results),
        successes,
        failures,
        skips,
    )

    if args.manifest:
        write_manifest(args.manifest, results)
        LOGGER.info("Wrote manifest to %s", args.manifest)

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
