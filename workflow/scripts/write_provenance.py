#!/usr/bin/env python
"""Write one provenance record for a finished batch job of alphafold3_datafill_predictions_batched.

Called at the end of each batch job on the GPU node (outside the container). Writes
alphafold3_predictions/.provenance/<job key>.json, where the job key is SLURM_JOB_ID (or local-<host>-<epoch>).
One file per job with one writer: parallel jobs never write the same file. The file is
written to <name>.tmp and renamed, so a reader never sees a partial record.

The record holds, for the job: SLURM ids, node, GPUs (host nvidia-smi), container path with its
labels and the sha256 from a `<image>.sha256` file next to it (if present; the image is not
hashed here), batch-infer commit, wrapper sha256, config.yaml sha256, the prediction settings,
and the FILL/CONTAINER timestamps. For each pool of the batch: its status (POOL_STATUS), tokens and
predicted runtime from the batch list, and any `[<tag>] ITEM name=<id> k=v ...` / `PHASE item=<id> ...`
lines the engine printed (the af3_torch kit does).

Never fails the job: errors are reported on stderr and the script exits 0.
Stdlib only.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

SCHEMA = "batch-infer-provenance/1"
KV_RE = re.compile(r"(\w+)=(\S+)")
ENGINE_RE = re.compile(r"^\[([\w.-]+)\] ([A-Z_]+) (.*)$")
POOL_RE = re.compile(r"^POOL_STATUS (\S+) (\S+)")


def sha256_file(path: Path) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def number(v: str):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return None if v == "None" else v


def kv(text: str) -> dict:
    return {k: number(v) for k, v in KV_RE.findall(text)}


def run(cmd: list[str]) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=True).stdout
    except Exception:
        return None


def gpus() -> list[dict]:
    out = run(["nvidia-smi", "--query-gpu=name,uuid,memory.total,compute_cap,driver_version,pci.bus_id",
               "--format=csv,noheader,nounits"])
    keys = ["name", "uuid", "memory_total_mib", "compute_cap", "driver", "pci_bus_id"]
    return [dict(zip(keys, (number(x.strip()) for x in line.split(","))))
            for line in (out or "").splitlines() if line.strip()]


def container_info(path: str) -> dict:
    p = Path(path)
    info = {"path": str(p.resolve()) if p.exists() else path, "size": p.stat().st_size if p.exists() else None}
    side = p.with_name(p.name + ".sha256")          # `sha256sum <image> > <image>.sha256`
    info["sha256"] = side.read_text().split()[0] if side.is_file() else None
    src = p.with_name(p.name + ".source")
    info["source_uri"] = src.read_text().strip() if src.is_file() else None
    labels = None
    for exe in ("singularity", "apptainer"):
        out = run([exe, "inspect", "--json", "--labels", path])
        if out:
            try:
                d = json.loads(out)
                labels = d.get("data", {}).get("attributes", {}).get("labels", d)
            except ValueError:
                labels = None
            break
    info["labels"] = labels
    if isinstance(labels, dict):
        info["oci_digest"] = labels.get("org.opencontainers.image.base.digest")
        info["oci_revision"] = labels.get("org.opencontainers.image.revision")
    return info


def git_state(root: Path) -> dict:
    head = run(["git", "-C", str(root), "rev-parse", "HEAD"])
    dirty = run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"])
    return {"root": str(root), "commit": head.strip() if head else None,
            "dirty": None if dirty is None else bool(dirty.strip())}


def parse_engine(log: Path, ids: set[str]) -> tuple[dict, dict]:
    """Per-pool engine lines (ITEM/PHASE keyed by name=/item=) and job-level ones (first of each kind)."""
    per_pool: dict[str, dict] = {}
    job: dict[str, dict] = {}
    if not log.is_file():
        return per_pool, job
    for line in log.read_text(errors="replace").splitlines():
        m = ENGINE_RE.match(line.strip())
        if not m:
            continue
        tag, kind, rest = m.groups()
        d = kv(rest)
        pid = d.get("name", d.get("item"))
        if isinstance(pid, str) and pid in ids:
            per_pool.setdefault(pid, {})[kind.lower()] = d
        elif kind not in ("COMMAND", "LEVER", "FALLBACK"):
            job.setdefault(f"{tag} {kind}", d)
    return per_pool, job


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--rule", required=True)
    ap.add_argument("--batch-id", required=True)
    ap.add_argument("--ids", required=True, help="space-separated pool ids of the batch")
    ap.add_argument("--batches-tsv", required=True, type=Path)
    ap.add_argument("--container", required=True)
    ap.add_argument("--wrapper", required=True, type=Path, help="host path of run_alphafold_wrapper")
    ap.add_argument("--settings", required=True, help="JSON: the alphafold3 config section of this job")
    ap.add_argument("--code-root", required=True, type=Path)
    ap.add_argument("--container-log", type=Path, help="tee'd container stdout/stderr")
    ap.add_argument("--pool-status", type=Path, help="POOL_STATUS lines of this job")
    ap.add_argument("--times", default="", help="k=v ... (job_start, fill_start, fill_end, container_start, container_end)")
    ap.add_argument("--container-rc", type=int)
    a = ap.parse_args()

    ids = a.ids.split()
    host = socket.gethostname()
    job_id = os.environ.get("SLURM_JOB_ID")
    key = job_id or f"local-{host}-{int(time.time())}"
    out_dir = a.run_dir / "alphafold3_predictions" / ".provenance"
    out_dir.mkdir(parents=True, exist_ok=True)

    batch = {}
    try:
        import csv
        with open(a.batches_tsv) as fh:
            batch = {r["id"]: r for r in csv.DictReader(fh, delimiter="\t")}
    except OSError:
        pass
    status = {}
    if a.pool_status and a.pool_status.is_file():
        for line in a.pool_status.read_text().splitlines():
            if m := POOL_RE.match(line.strip()):
                status[m.group(1)] = m.group(2)
    engine_pool, engine_job = parse_engine(a.container_log, set(ids)) if a.container_log else ({}, {})

    pools = []
    for pid in ids:
        b = batch.get(pid, {})
        pools.append({"id": pid, "status": status.get(pid),
                      "tokens": number(b["tokens"]) if b.get("tokens") else None,
                      "est_runtime_s": number(b["runtime_s"]) if b.get("runtime_s") else None,
                      "engine": engine_pool.get(pid, {})})

    config_yaml = a.run_dir / "config.yaml"
    rec = {
        "schema": SCHEMA,
        "source": "batch-infer",
        "written": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_dir": str(a.run_dir.resolve()),
        "rule": a.rule,
        "batch_id": a.batch_id,
        "slurm": {k.removeprefix("SLURM_").lower(): v for k, v in os.environ.items()
                  if k in ("SLURM_JOB_ID", "SLURM_CLUSTER_NAME", "SLURM_JOB_PARTITION", "SLURM_JOB_NODELIST",
                           "SLURM_JOB_NAME", "SLURM_SUBMIT_HOST", "SLURM_RESTART_COUNT", "SLURM_JOB_GPUS")},
        "host": host,
        "gpus": gpus(),
        "container": container_info(a.container),
        "code": {"batch_infer": git_state(a.code_root),
                 "wrapper": {"path": str(a.wrapper), "sha256": sha256_file(a.wrapper)}},
        "config": {"config_yaml_sha256": sha256_file(config_yaml) if config_yaml.is_file() else None,
                   "alphafold3": json.loads(a.settings)},
        "times": kv(a.times),
        "container_rc": a.container_rc,
        "engine": engine_job,
        "pools": pools,
    }
    out = out_dir / f"{key}.json"
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec, indent=1, default=str) + "\n")
    os.replace(tmp, out)
    n_ok = sum(p["status"] == "ok" for p in pools)
    print(f"PROVENANCE: {out} ({n_ok} of {len(pools)} pools ok)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:      # provenance must never fail a batch that produced zips
        print(f"PROVENANCE_FAILED: {type(e).__name__}: {e}", file=sys.stderr)
