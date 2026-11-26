"""Resource detection: GPU VRAM, system RAM, container (cgroup) limits, CPU count.

Inside a container ``psutil`` reports the *host's* memory, so the effective memory is
``min(host RAM, cgroup limit, explicit hint)``. The hint comes from the environment
variables ``SAPIA_MEM_LIMIT_MB`` / ``SAPIA_CPU_LIMIT``; a container launcher can set them to
the limits it requests, so the numbers are right even if a runtime hides its cgroup files.

Run ``python -m sapia.resources`` to print what is detected on this machine.
"""
from __future__ import annotations

import math
import os
import platform
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from .utils import fmt_bytes

# cgroup v1 reports "no limit" as a huge page-aligned number (typically 0x7FFFFFFFFFFFF000).
UNLIMITED_THRESHOLD = 1 << 60

ENV_MEM_LIMIT_MB = "SAPIA_MEM_LIMIT_MB"
ENV_CPU_LIMIT = "SAPIA_CPU_LIMIT"


# ----------------------------------------------------------------------------- cgroups

@dataclass
class CgroupMemory:
    version: Optional[int] = None          # 1, 2 or None when no cgroup memory controller is visible
    limit_bytes: Optional[int] = None      # None = unlimited / unknown
    usage_bytes: Optional[int] = None      # current usage, including page cache
    inactive_file_bytes: Optional[int] = None  # reclaimable page cache

    @property
    def available_bytes(self) -> Optional[int]:
        """Limit minus non-reclaimable usage, or None when there is no limit."""
        if self.limit_bytes is None or self.usage_bytes is None:
            return None
        used = self.usage_bytes - (self.inactive_file_bytes or 0)
        return max(0, self.limit_bytes - max(0, used))


def _read_text(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except (OSError, ValueError):
        return None


def _parse_limit(text: Optional[str]) -> Optional[int]:
    """Parse a cgroup limit file: an integer, or ``max`` / a huge sentinel for unlimited."""
    if text is None:
        return None
    token = text.split()[0] if text.split() else ""
    if token in ("", "max"):
        return None
    try:
        value = int(token)
    except ValueError:
        return None
    if value <= 0 or value >= UNLIMITED_THRESHOLD:
        return None
    return value


def _parse_int(text: Optional[str]) -> Optional[int]:
    if text is None:
        return None
    try:
        return int(text.split()[0])
    except (ValueError, IndexError):
        return None


def _parse_stat(text: Optional[str], key: str) -> Optional[int]:
    if text is None:
        return None
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] == key:
            try:
                return int(parts[1])
            except ValueError:
                return None
    return None


def _own_cgroup_paths(proc_cgroup: Path) -> tuple[Optional[str], dict[str, str]]:
    """Parse /proc/self/cgroup -> (v2 relative path, {v1 controller: relative path})."""
    v2: Optional[str] = None
    v1: dict[str, str] = {}
    text = _read_text(proc_cgroup)
    if not text:
        return v2, v1
    for line in text.splitlines():
        parts = line.strip().split(":", 2)
        if len(parts) != 3:
            continue
        hierarchy, controllers, path = parts
        if hierarchy == "0" and controllers == "":
            v2 = path
        else:
            for controller in controllers.split(","):
                if controller:
                    v1[controller] = path
    return v2, v1


def _candidate_dirs(base: Path, rel: Optional[str]) -> list[Path]:
    """``base/rel`` and each of its ancestors down to ``base`` (deepest first).

    In a cgroup-namespaced container ``rel`` is "/" and only ``base`` is returned. When
    the path from /proc/self/cgroup is a host path that is not visible, only the
    directories that exist are kept (``base`` always is).
    """
    dirs: list[Path] = []
    if rel:
        parts = [p for p in rel.split("/") if p and p != ".."]
        for i in range(len(parts), 0, -1):
            dirs.append(base.joinpath(*parts[:i]))
    dirs.append(base)
    return [d for d in dirs if d.is_dir()]


def read_cgroup_memory(cgroup_root: str | os.PathLike = "/sys/fs/cgroup",
                       proc_cgroup: str | os.PathLike = "/proc/self/cgroup") -> CgroupMemory:
    """Read the memory limit and usage that apply to this process (cgroup v2, then v1).

    The effective limit is the smallest limit among this process's cgroup and its
    ancestors; usage is read from the cgroup that imposes that limit.
    """
    root = Path(cgroup_root)
    if not root.is_dir():
        return CgroupMemory()
    v2_rel, v1_rel = _own_cgroup_paths(Path(proc_cgroup))

    # ---- cgroup v2 (unified hierarchy)
    if (root / "cgroup.controllers").exists() or (root / "memory.max").exists():
        best: Optional[tuple[int, Path]] = None
        seen = False
        for d in _candidate_dirs(root, v2_rel):
            if (d / "memory.max").exists() or (d / "memory.current").exists():
                seen = True
            limit = _parse_limit(_read_text(d / "memory.max"))
            if limit is not None and (best is None or limit < best[0]):
                best = (limit, d)
        if best is not None:
            limit, d = best
            return CgroupMemory(
                version=2, limit_bytes=limit,
                usage_bytes=_parse_int(_read_text(d / "memory.current")),
                inactive_file_bytes=_parse_stat(_read_text(d / "memory.stat"), "inactive_file"),
            )
        if seen or (root / "cgroup.controllers").exists():
            return CgroupMemory(version=2)

    # ---- cgroup v1 (memory controller mounted at <root>/memory)
    mem_root = root / "memory"
    if mem_root.is_dir():
        best = None
        for d in _candidate_dirs(mem_root, v1_rel.get("memory")):
            limit = _parse_limit(_read_text(d / "memory.limit_in_bytes"))
            if limit is not None and (best is None or limit < best[0]):
                best = (limit, d)
        if best is not None:
            limit, d = best
            stat = _read_text(d / "memory.stat")
            inactive = _parse_stat(stat, "total_inactive_file")
            if inactive is None:
                inactive = _parse_stat(stat, "inactive_file")
            return CgroupMemory(
                version=1, limit_bytes=limit,
                usage_bytes=_parse_int(_read_text(d / "memory.usage_in_bytes")),
                inactive_file_bytes=inactive,
            )
        return CgroupMemory(version=1)

    return CgroupMemory()


def read_cgroup_cpu_quota(cgroup_root: str | os.PathLike = "/sys/fs/cgroup",
                          proc_cgroup: str | os.PathLike = "/proc/self/cgroup") -> Optional[float]:
    """CPU quota in cores (e.g. 4.0) from cgroup v2 ``cpu.max`` or v1 cfs files; None if unlimited."""
    root = Path(cgroup_root)
    if not root.is_dir():
        return None
    v2_rel, v1_rel = _own_cgroup_paths(Path(proc_cgroup))
    quotas: list[float] = []

    for d in _candidate_dirs(root, v2_rel):
        text = _read_text(d / "cpu.max")
        if text:
            parts = text.split()
            if len(parts) == 2 and parts[0] != "max":
                try:
                    quota, period = int(parts[0]), int(parts[1])
                    if quota > 0 and period > 0:
                        quotas.append(quota / period)
                except ValueError:
                    pass

    for name in ("cpu", "cpu,cpuacct", "cpuacct,cpu"):
        cpu_root = root / name
        if not cpu_root.is_dir():
            continue
        rel = v1_rel.get("cpu") or v1_rel.get("cpuacct")
        for d in _candidate_dirs(cpu_root, rel):
            quota = _parse_int(_read_text(d / "cpu.cfs_quota_us"))
            period = _parse_int(_read_text(d / "cpu.cfs_period_us"))
            if quota is not None and period and quota > 0:
                quotas.append(quota / period)
        break

    return min(quotas) if quotas else None


# ----------------------------------------------------------------------------- host memory / cpu

def _host_memory() -> tuple[int, int]:
    """(total, available) physical memory in bytes, as the OS reports it."""
    try:
        import psutil

        vm = psutil.virtual_memory()
        return int(vm.total), int(vm.available)
    except Exception:
        pass
    meminfo = _read_text(Path("/proc/meminfo"))
    if meminfo:
        values = {}
        for line in meminfo.splitlines():
            parts = line.replace(":", " ").split()
            if len(parts) >= 2 and parts[1].isdigit():
                values[parts[0]] = int(parts[1]) * 1024
        if "MemTotal" in values:
            return values["MemTotal"], values.get("MemAvailable", values.get("MemFree", values["MemTotal"]))
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        return int(total), int(total)
    except (AttributeError, ValueError, OSError):
        return 0, 0


def _affinity_count() -> Optional[int]:
    try:
        return len(os.sched_getaffinity(0))  # Linux only
    except (AttributeError, OSError):
        return None


def _env_float(name: str, env: Optional[dict] = None) -> Optional[float]:
    raw = (env if env is not None else os.environ).get(name)
    if raw is None or str(raw).strip() == "":
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if value > 0 else None


# ----------------------------------------------------------------------------- GPU

@dataclass
class GPUInfo:
    index: int
    name: str
    total_bytes: int
    free_bytes: int
    capability: tuple[int, int]
    bf16_supported: bool


def detect_gpus() -> list[GPUInfo]:
    try:
        import torch
    except Exception:
        return []
    if not torch.cuda.is_available():
        return []
    gpus: list[GPUInfo] = []
    for i in range(torch.cuda.device_count()):
        try:
            props = torch.cuda.get_device_properties(i)
            free, total = torch.cuda.mem_get_info(i)
            major, minor = torch.cuda.get_device_capability(i)
            try:
                with torch.cuda.device(i):
                    bf16 = bool(torch.cuda.is_bf16_supported())
            except Exception:
                bf16 = False
            gpus.append(GPUInfo(i, props.name, int(total), int(free), (int(major), int(minor)), bf16))
        except Exception:
            continue
    return gpus


# ----------------------------------------------------------------------------- aggregate

@dataclass
class Resources:
    platform: str
    cpu_logical: int                 # os.cpu_count()
    cpu_usable: int                  # after affinity, cgroup quota and env hint
    cpu_quota: Optional[float]       # cgroup CPU quota in cores, if any
    host_mem_total_bytes: int        # what psutil / the OS reports
    host_mem_available_bytes: int
    cgroup: CgroupMemory
    hint_mem_limit_bytes: Optional[int]  # from SAPIA_MEM_LIMIT_MB
    mem_total_bytes: int             # effective: min(host, cgroup limit, hint)
    mem_available_bytes: int         # effective: min(host available, limit - usage)
    mem_limit_source: str            # "host" | "cgroup v1" | "cgroup v2" | "env hint"
    gpus: list[GPUInfo] = field(default_factory=list)
    in_container: bool = False

    @property
    def gpu(self) -> Optional[GPUInfo]:
        return self.gpus[0] if self.gpus else None

    def to_dict(self) -> dict:
        return asdict(self)

    def summary_lines(self) -> list[str]:
        lines = [
            f"platform        : {self.platform}{' (container)' if self.in_container else ''}",
            f"cpu             : {self.cpu_usable} usable of {self.cpu_logical} logical"
            + (f" (cgroup quota {self.cpu_quota:.2f} cores)" if self.cpu_quota else ""),
            f"host RAM        : {fmt_bytes(self.host_mem_total_bytes)} total, "
            f"{fmt_bytes(self.host_mem_available_bytes)} available",
        ]
        if self.cgroup.version is not None:
            lines.append(
                f"cgroup v{self.cgroup.version} memory: limit {fmt_bytes(self.cgroup.limit_bytes) if self.cgroup.limit_bytes else 'none'}"
                f", usage {fmt_bytes(self.cgroup.usage_bytes)}"
            )
        if self.hint_mem_limit_bytes:
            lines.append(f"memory hint     : {fmt_bytes(self.hint_mem_limit_bytes)} (from {ENV_MEM_LIMIT_MB})")
        lines.append(
            f"effective RAM   : {fmt_bytes(self.mem_total_bytes)} total, "
            f"{fmt_bytes(self.mem_available_bytes)} available (limit source: {self.mem_limit_source})"
        )
        if not self.gpus:
            lines.append("gpu             : none (CPU only)")
        for g in self.gpus:
            lines.append(
                f"gpu {g.index}           : {g.name}, {fmt_bytes(g.total_bytes)} total, {fmt_bytes(g.free_bytes)} free, "
                f"capability {g.capability[0]}.{g.capability[1]}, bf16 {'yes' if g.bf16_supported else 'no'}"
            )
        return lines


def combine_memory(host_total: int, host_available: int, cgroup: CgroupMemory,
                   hint_limit: Optional[int]) -> tuple[int, int, str]:
    """Effective (total, available, source) from host numbers, cgroup limits and the env hint."""
    host_known = host_total > 0
    total, source = host_total, "host"
    candidates = [host_available] if host_known else []

    if cgroup.limit_bytes is not None and (not host_known or cgroup.limit_bytes < total):
        total, source = cgroup.limit_bytes, f"cgroup v{cgroup.version}"
    if cgroup.available_bytes is not None:
        candidates.append(cgroup.available_bytes)

    if hint_limit is not None and (total <= 0 or hint_limit < total):
        total, source = hint_limit, "env hint"
        # A hint carries no usage accounting; subtract the cgroup's non-reclaimable usage
        # when it is known, otherwise assume the container is still (nearly) empty.
        used = 0
        if cgroup.usage_bytes is not None:
            used = max(0, cgroup.usage_bytes - (cgroup.inactive_file_bytes or 0))
        candidates.append(max(0, hint_limit - used))

    available = min(candidates) if candidates else total
    available = max(0, min(available, total))
    return int(total), int(available), source


def detect_resources(cgroup_root: str | os.PathLike = "/sys/fs/cgroup",
                     proc_cgroup: str | os.PathLike = "/proc/self/cgroup",
                     env: Optional[dict] = None, with_gpu: bool = True) -> Resources:
    host_total, host_available = _host_memory()
    cgroup = read_cgroup_memory(cgroup_root, proc_cgroup)
    hint_mb = _env_float(ENV_MEM_LIMIT_MB, env)
    hint_limit = int(hint_mb * 1024 * 1024) if hint_mb else None
    total, available, source = combine_memory(host_total, host_available, cgroup, hint_limit)

    logical = os.cpu_count() or 1
    usable = float(logical)
    affinity = _affinity_count()
    if affinity:
        usable = min(usable, affinity)
    quota = read_cgroup_cpu_quota(cgroup_root, proc_cgroup)
    if quota:
        usable = min(usable, quota)
    cpu_hint = _env_float(ENV_CPU_LIMIT, env)
    if cpu_hint:
        usable = min(usable, cpu_hint)
    cpu_usable = max(1, int(math.floor(usable + 1e-9)))

    environ = env if env is not None else os.environ
    in_container = bool(
        cgroup.limit_bytes is not None
        or environ.get("MODAL_TASK_ID")
        or environ.get("KUBERNETES_SERVICE_HOST")
        or Path("/.dockerenv").exists()
    )
    return Resources(
        platform=f"{platform.system()} {platform.release()}",
        cpu_logical=logical, cpu_usable=cpu_usable, cpu_quota=quota,
        host_mem_total_bytes=host_total, host_mem_available_bytes=host_available,
        cgroup=cgroup, hint_mem_limit_bytes=hint_limit,
        mem_total_bytes=total, mem_available_bytes=available, mem_limit_source=source,
        gpus=detect_gpus() if with_gpu else [], in_container=in_container,
    )


def main() -> None:
    res = detect_resources()
    for line in res.summary_lines():
        print(line)


if __name__ == "__main__":
    main()
