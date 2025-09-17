from __future__ import annotations

import argparse
import json
import logging
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from src.utils.io import ensure_dirs

LOGGER = logging.getLogger(__name__)

try:  # tqdm is optional; fall back to simple progress if it is missing
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None


class _DummyProgress:
    def update(self, n: int = 1) -> None:  # noqa: D401 - trivial helper
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


@dataclass
class DecodeResult:
    filename: str
    status: str
    returncode: Optional[int]
    duration_sec: float
    output_dir: Path
    stdout: Optional[str]
    stderr: Optional[str]
    sample_id: Optional[str]
    subject_id: Optional[str]
    sample_type: Optional[str]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decode MPEG-G .mgb files into analysis-ready reads")
    parser.add_argument(
        "--mgb-dir",
        type=Path,
        default=Path("data/TrainFiles"),
        help="Directory containing .mgb files to decode",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/decoded"),
        help="Where decoded outputs should be written (per-file subdirectories)",
    )
    parser.add_argument(
        "--decoder-bin",
        type=str,
        default="mpeg-g",
        help="Executable used to perform decoding",
    )
    parser.add_argument(
        "--command-template",
        type=str,
        default="{decoder} decode --input {input} --output-dir {output_dir}",
        help=(
            "Template for the decode command. Use placeholders {decoder}, {input}, {output_dir}, {stem}, {output_root}. "
            "Example: 'mpeg-g extract --in {input} --out {output_dir} --format fastq'."
        ),
    )
    parser.add_argument(
        "--train-index",
        type=Path,
        default=Path("data/Train.csv"),
        help="CSV mapping filename to sample metadata (optional but recommended)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of concurrent decode processes to run",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional path to write a JSONL manifest summarizing decode results",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not rerun decoding when the output directory already exists and is non-empty",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress reporting (useful for CI logs)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity",
    )
    return parser.parse_args(argv)


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


def discover_mgb_files(mgb_dir: Path) -> List[Path]:
    if not mgb_dir.exists():
        raise FileNotFoundError(f"Input directory {mgb_dir} does not exist")
    files = sorted(mgb_dir.glob("*.mgb"))
    if not files:
        LOGGER.warning("No .mgb files found under %s", mgb_dir)
    return files


def build_command(
    template: str,
    decoder_bin: str,
    mgb_path: Path,
    output_root: Path,
    per_file_dir: Path,
) -> List[str]:
    cmd_str = template.format(
        decoder=decoder_bin,
        input=str(mgb_path),
        output_dir=str(per_file_dir),
        stem=mgb_path.stem,
        output_root=str(output_root),
    )
    return shlex.split(cmd_str)


def decode_one(
    mgb_path: Path,
    output_root: Path,
    decoder_bin: str,
    template: str,
    skip_existing: bool,
) -> DecodeResult:
    per_file_dir = output_root / mgb_path.stem
    ensure_dirs(per_file_dir)

    if skip_existing and any(per_file_dir.iterdir()):
        LOGGER.info("Skipping %s (outputs already exist)", mgb_path.name)
        return DecodeResult(
            filename=mgb_path.name,
            status="skipped",
            returncode=None,
            duration_sec=0.0,
            output_dir=per_file_dir,
            stdout=None,
            stderr=None,
            sample_id=None,
            subject_id=None,
            sample_type=None,
        )

    cmd = build_command(template, decoder_bin, mgb_path, output_root, per_file_dir)
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
        raise FileNotFoundError(f"Decoder executable not found: {cmd[0]}") from exc

    status = "success" if proc.returncode == 0 else "failed"
    if status == "failed":
        LOGGER.error("Decoding %s failed with return code %s", mgb_path.name, proc.returncode)
        LOGGER.debug("stderr: %s", proc.stderr)

    return DecodeResult(
        filename=mgb_path.name,
        status=status,
        returncode=proc.returncode,
        duration_sec=duration,
        output_dir=per_file_dir,
        stdout=proc.stdout.strip() or None,
        stderr=proc.stderr.strip() or None,
        sample_id=None,
        subject_id=None,
        sample_type=None,
    )


def enrich_with_metadata(results: Iterable[DecodeResult], metadata: Dict[str, Dict[str, str]]) -> List[DecodeResult]:
    enriched: List[DecodeResult] = []
    for res in results:
        meta = metadata.get(res.filename)
        if meta:
            enriched.append(
                DecodeResult(
                    filename=res.filename,
                    status=res.status,
                    returncode=res.returncode,
                    duration_sec=res.duration_sec,
                    output_dir=res.output_dir,
                    stdout=res.stdout,
                    stderr=res.stderr,
                    sample_id=meta.get("SampleID"),
                    subject_id=meta.get("SubjectID"),
                    sample_type=meta.get("SampleType"),
                )
            )
        else:
            enriched.append(res)
    return enriched


def write_manifest(path: Path, results: Iterable[DecodeResult]) -> None:
    ensure_dirs(path.parent)
    with path.open("w", encoding="utf-8") as fh:
        for res in results:
            record = {
                "filename": res.filename,
                "status": res.status,
                "returncode": res.returncode,
                "duration_sec": round(res.duration_sec, 3),
                "output_dir": str(res.output_dir),
                "stdout": res.stdout,
                "stderr": res.stderr,
                "SampleID": res.sample_id,
                "SubjectID": res.subject_id,
                "SampleType": res.sample_type,
            }
            fh.write(json.dumps(record) + "\n")


def run_decodes(
    mgb_files: Iterable[Path],
    output_root: Path,
    decoder_bin: str,
    template: str,
    skip_existing: bool,
    workers: int,
    show_progress: bool,
) -> List[DecodeResult]:
    mgb_list = list(mgb_files)
    total = len(mgb_list)
    if total == 0:
        return []

    if show_progress:
        if tqdm is not None:
            progress = _TqdmProgress(total, "Decoding")
        else:
            progress = _SimpleProgress(total, "Decoding")
    else:
        progress = _DummyProgress()

    results: List[DecodeResult] = []

    if workers <= 1:
        for mgb in mgb_list:
            results.append(
                decode_one(mgb, output_root, decoder_bin, template, skip_existing)
            )
            progress.update()
        progress.close()
        return results

    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {
            pool.submit(
                decode_one,
                mgb,
                output_root,
                decoder_bin,
                template,
                skip_existing,
            ): mgb
            for mgb in mgb_list
        }
        for future in as_completed(future_map):
            try:
                results.append(future.result())
            except Exception as exc:  # pragma: no cover - defensive logging
                mgb = future_map[future]
                LOGGER.exception("Decoding %s raised an error", mgb.name)
                results.append(
                    DecodeResult(
                        filename=mgb.name,
                        status="error",
                        returncode=None,
                        duration_sec=0.0,
                        output_dir=output_root / mgb.stem,
                        stdout=None,
                        stderr=str(exc),
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

    mgb_files = discover_mgb_files(args.mgb_dir)
    if not mgb_files:
        return 0

    ensure_dirs(args.output_root)
    metadata = load_train_metadata(args.train_index) if args.train_index else {}

    results = run_decodes(
        mgb_files=mgb_files,
        output_root=args.output_root,
        decoder_bin=args.decoder_bin,
        template=args.command_template,
        skip_existing=args.skip_existing,
        workers=args.workers,
        show_progress=not args.no_progress,
    )

    results = enrich_with_metadata(results, metadata)

    successes = sum(r.status == "success" for r in results)
    failures = sum(r.status == "failed" for r in results)
    skips = sum(r.status == "skipped" for r in results)
    errors = sum(r.status == "error" for r in results)
    LOGGER.info(
        "Decoded %s files (%s success, %s failed, %s skipped, %s errors)",
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
