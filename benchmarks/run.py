#!/usr/bin/env python3
"""Run Rechenwerk/PyTorch MNIST benchmarks and print a Markdown table."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


EXAMPLES = ("mnist_mlp", "mnist_cnn", "mnist_vit")
FRAMEWORK_ORDER = {"Rechenwerk": 0, "PyTorch": 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--rew-backend", default="cpu")
    parser.add_argument("--torch-device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=0)
    parser.add_argument("--skip-rew", action="store_true")
    parser.add_argument("--skip-pytorch", action="store_true")
    return parser.parse_args()


def add_failed_rows(
    rows: list[dict[str, Any]],
    framework: str,
    device: str,
    args: argparse.Namespace,
    reason: str,
) -> None:
    for example in EXAMPLES:
        rows.append(
            {
                "example": example,
                "framework": framework,
                "device": device,
                "status": "failed",
                "reason": reason,
                "batch_size": args.batch_size,
                "warmup": args.warmup,
                "iterations": args.iterations,
                "first_step_ms": None,
                "mean_step_ms": None,
                "samples_per_s": None,
                "last_loss": None,
            }
        )


def add_skipped_rows(
    rows: list[dict[str, Any]],
    framework: str,
    device: str,
    args: argparse.Namespace,
    reason: str,
) -> None:
    for example in EXAMPLES:
        rows.append(
            {
                "example": example,
                "framework": framework,
                "device": device,
                "status": "skipped",
                "reason": reason,
                "batch_size": args.batch_size,
                "warmup": args.warmup,
                "iterations": args.iterations,
                "first_step_ms": None,
                "mean_step_ms": None,
                "samples_per_s": None,
                "last_loss": None,
            }
        )


def parse_rows(output: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError:
                pass
    return rows


def run_command(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def run_rew(root: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if args.skip_rew:
        add_skipped_rows(rows, "Rechenwerk", args.rew_backend, args, "disabled")
        return rows

    out_bin = root / "benchmarks" / "rew_mnist_bench"
    compile_cmd = [
        "nim",
        "c",
        "-d:release",
        "--hints:off",
        f"--out:{out_bin}",
        str(root / "benchmarks" / "rew_mnist.nim"),
    ]
    compiled = run_command(compile_cmd, root)
    if compiled.returncode != 0:
        reason = "Nim benchmark failed to compile"
        if compiled.stdout.strip():
            reason += ": " + compiled.stdout.strip().splitlines()[-1]
        add_failed_rows(rows, "Rechenwerk", args.rew_backend, args, reason)
        return rows

    run_cmd = [
        str(out_bin),
        f"--backend={args.rew_backend}",
        f"--warmup={args.warmup}",
        f"--iterations={args.iterations}",
        f"--batch-size={args.batch_size}",
    ]
    completed = run_command(run_cmd, root)
    rows.extend(parse_rows(completed.stdout))
    if completed.returncode != 0 and not rows:
        reason = "Nim benchmark failed"
        if completed.stdout.strip():
            reason += ": " + completed.stdout.strip().splitlines()[-1]
        add_failed_rows(rows, "Rechenwerk", args.rew_backend, args, reason)
    return rows


def run_pytorch(root: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if args.skip_pytorch:
        add_skipped_rows(rows, "PyTorch", args.torch_device, args, "disabled")
        return rows

    command = [
        sys.executable,
        str(root / "benchmarks" / "pytorch_mnist.py"),
        f"--device={args.torch_device}",
        f"--warmup={args.warmup}",
        f"--iterations={args.iterations}",
        f"--batch-size={args.batch_size}",
        f"--threads={args.torch_threads}",
    ]
    completed = run_command(command, root)
    rows.extend(parse_rows(completed.stdout))
    if completed.returncode != 0 and not rows:
        reason = "PyTorch benchmark failed"
        if completed.stdout.strip():
            reason += ": " + completed.stdout.strip().splitlines()[-1]
        add_failed_rows(rows, "PyTorch", args.torch_device, args, reason)
    return rows


def fmt_number(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def fmt_status(row: dict[str, Any]) -> str:
    status = str(row.get("status", "unknown"))
    reason = str(row.get("reason", ""))
    if status == "ok" or not reason:
        return status
    reason = " ".join(reason.split())
    if len(reason) > 86:
        reason = reason[:83] + "..."
    return f"{status}: {reason}"


def sorted_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    example_order = {name: index for index, name in enumerate(EXAMPLES)}
    return sorted(
        rows,
        key=lambda row: (
            example_order.get(str(row.get("example")), 99),
            FRAMEWORK_ORDER.get(str(row.get("framework")), 99),
            str(row.get("framework")),
        ),
    )


def print_table(rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    print("| Example | Framework | Device | Batch | First step ms | Mean step ms | Samples/s | Last loss | Status |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---|")
    for row in sorted_rows(rows):
        print(
            "| {example} | {framework} | {device} | {batch} | {first} | "
            "{mean} | {samples} | {loss} | {status} |".format(
                example=row.get("example", "-"),
                framework=row.get("framework", "-"),
                device=row.get("device", "-"),
                batch=row.get("batch_size", args.batch_size),
                first=fmt_number(row.get("first_step_ms")),
                mean=fmt_number(row.get("mean_step_ms")),
                samples=fmt_number(row.get("samples_per_s"), digits=1),
                loss=fmt_number(row.get("last_loss"), digits=4),
                status=fmt_status(row),
            )
        )
    print()
    print(
        "First step includes tracing/compilation for Rechenwerk. "
        "Mean step excludes the first step and warmup iterations."
    )


def main() -> int:
    args = parse_args()
    if args.iterations <= 0 or args.warmup < 0 or args.batch_size <= 0:
        raise SystemExit(
            "warmup must be non-negative; iterations and batch-size must be positive"
        )

    root = Path(__file__).resolve().parents[1]
    rows: list[dict[str, Any]] = []
    rows.extend(run_rew(root, args))
    rows.extend(run_pytorch(root, args))

    print_table(rows, args)
    return 1 if any(row.get("status") == "failed" for row in rows) else 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    raise SystemExit(main())
