"""GPU selection helpers for a shared machine.

This box is shared with other users. Starting a job on a GPU somebody else is
already using risks OOM-ing both, so selection keys on *process ownership*
rather than raw free memory: another user's process blocks the GPU, while this
user's own processes do not.
"""

from __future__ import annotations

import getpass
import os
import subprocess
from typing import Dict, List, Optional, Sequence, Tuple


def _smi(query: str, extra: Sequence[str] = ()) -> List[List[str]]:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-{query}", "--format=csv,noheader,nounits", *extra],
            capture_output=True, text=True, timeout=30, check=True).stdout
    except Exception:
        return []
    return [[c.strip() for c in line.split(",")]
            for line in out.strip().splitlines() if line.strip()]


def gpu_memory_used() -> Dict[int, int]:
    """Per-GPU memory in use, in MiB."""
    return {int(idx): int(mib) for idx, mib in _smi("gpu=index,memory.used")}


def _uuid_to_index() -> Dict[str, int]:
    return {uuid: int(idx) for idx, uuid in _smi("gpu=index,uuid")}


def _process_owner(pid: str) -> str:
    try:
        return subprocess.run(["ps", "-o", "user=", "-p", pid],
                              capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:
        return ""


def foreign_usage(user: Optional[str] = None) -> Dict[int, List[Tuple[str, str, int]]]:
    """Per-GPU list of (pid, user, MiB) for processes owned by somebody else."""
    me = user or getpass.getuser()
    idx_of = _uuid_to_index()
    out: Dict[int, List[Tuple[str, str, int]]] = {}
    for row in _smi("compute-apps=pid,gpu_uuid,used_memory"):
        if len(row) < 3:
            continue
        pid, uuid, mib = row[0], row[1], row[2]
        owner = _process_owner(pid)
        if owner and owner != me:
            idx = idx_of.get(uuid)
            if idx is not None:
                out.setdefault(idx, []).append((pid, owner, int(mib)))
    return out


def free_gpus(threshold_mib: int = 2000) -> List[int]:
    """GPUs no other user is holding meaningful memory on."""
    foreign = foreign_usage()
    used = gpu_memory_used()
    return [g for g in sorted(used)
            if sum(m for _, _, m in foreign.get(g, [])) <= threshold_mib]


def check_gpus_free(gpus: Sequence[Optional[int]], threshold_mib: int = 2000) -> None:
    """Raise unless every requested GPU is free of other users' work."""
    wanted = [g for g in gpus if g is not None]
    if not wanted:
        return
    foreign = foreign_usage()
    if not foreign and not gpu_memory_used():
        return  # nvidia-smi unavailable; nothing to check against
    busy = []
    for g in wanted:
        procs = foreign.get(g, [])
        total = sum(m for _, _, m in procs)
        if total > threshold_mib:
            who = ", ".join(sorted({u for _, u, _ in procs}))
            busy.append(f"GPU {g}: {total:,} MiB held by {who}")
    if busy:
        raise SystemExit(
            "Refusing to start; another user is on the requested GPU(s):\n  "
            + "\n  ".join(busy)
            + f"\nFree now: {free_gpus(threshold_mib) or 'none'}"
            + "\nPass --force-gpus to override.")


def select_visible(gpu: Optional[int]) -> str:
    """Pin this process to one GPU and return the torch device string.

    Must be called before torch/transformers import CUDA.
    """
    if gpu is None:
        return "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return "cuda:0"
