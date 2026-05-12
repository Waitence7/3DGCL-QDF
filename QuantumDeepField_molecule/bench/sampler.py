"""Process tree + Windows GPU performance counter sampler.

Used by ``bench/run_pipeline.ipynb`` to compare NumPy and Rust variants of the
QDF pipeline on (wall time, RSS, CPU%, XPU%).

Design
------
* CPU and RSS are sampled per-process via ``psutil`` and aggregated over the
  whole subprocess tree (parent + descendants discovered each tick).
* XPU utilization on Windows is read through the PDH (Performance Data Helper)
  API. ``\\GPU Engine(*)\\Utilization Percentage`` returns one instance per
  (pid, engine) pair; instance names look like
  ``pid_10024_luid_0x00000000_0x0001779d_phys_0_eng_0_engtype_3d`` so we filter
  by the PIDs that belong to the sampled subprocess tree.
* Linux / macOS fall back to "no XPU samples available". CPU and RSS still work.

Output
------
``run_with_metrics`` returns ``(returncode, stdout_text, samples, summary)``.
``samples`` is a dict of equal-length lists ready to be wrapped in a DataFrame
or fed straight to matplotlib. ``summary`` contains scalar aggregates that the
notebook plots as bar charts.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence

import psutil

# --------------------------------------------------------------------------- #
# Windows PDH wrapper -- only loaded on Windows.                              #
# --------------------------------------------------------------------------- #

_IS_WINDOWS = os.name == "nt"

if _IS_WINDOWS:
    import ctypes
    from ctypes import wintypes

    _PDH = ctypes.WinDLL("pdh.dll")
    _PDH_FMT_DOUBLE = 0x00000200
    _PDH_MORE_DATA = 0x800007D2  # signed view: -2147481646
    _ERROR_SUCCESS = 0

    class _PDH_FMT_COUNTERVALUE(ctypes.Structure):
        _fields_ = [
            ("CStatus", wintypes.DWORD),
            # 4 bytes of padding before the 8-byte aligned union value.
            ("doubleValue", ctypes.c_double),
        ]

    class _PDH_FMT_COUNTERVALUE_ITEM_W(ctypes.Structure):
        _fields_ = [
            ("szName", wintypes.LPWSTR),
            ("FmtValue", _PDH_FMT_COUNTERVALUE),
        ]

    _PDH.PdhOpenQueryW.argtypes = [wintypes.LPCWSTR, ctypes.c_void_p, ctypes.POINTER(wintypes.HANDLE)]
    _PDH.PdhOpenQueryW.restype = wintypes.LONG
    _PDH.PdhAddEnglishCounterW.argtypes = [wintypes.HANDLE, wintypes.LPCWSTR, ctypes.c_void_p, ctypes.POINTER(wintypes.HANDLE)]
    _PDH.PdhAddEnglishCounterW.restype = wintypes.LONG
    _PDH.PdhCollectQueryData.argtypes = [wintypes.HANDLE]
    _PDH.PdhCollectQueryData.restype = wintypes.LONG
    _PDH.PdhCloseQuery.argtypes = [wintypes.HANDLE]
    _PDH.PdhCloseQuery.restype = wintypes.LONG
    _PDH.PdhGetFormattedCounterArrayW.argtypes = [
        wintypes.HANDLE, wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(wintypes.DWORD),
        ctypes.c_void_p,
    ]
    _PDH.PdhGetFormattedCounterArrayW.restype = wintypes.LONG


# Regex: instance name "pid_<digits>_..."  (used by \GPU Engine counters)
_PID_RE = re.compile(r"^pid_(\d+)_")

# Regex: instance name "<procname>:<pid>"  (used by \Process V2 counters)
_PROC_V2_RE = re.compile(r":(\d+)$")


# --------------------------------------------------------------------------- #
# Generic wildcard PDH counter helper                                         #
# --------------------------------------------------------------------------- #

class _PdhWildcardCounter:
    """Open + sample a single ``\\Object(*)\\Counter`` query.

    ``parse_instance`` extracts an integer (typically a PID) from the
    multi-instance counter's ``szName`` so callers can filter to a subset.
    Returns ``{key: summed_value}`` from :py:meth:`sample`.
    """

    def __init__(self, counter_path: str, parse_instance) -> None:
        self.available = False
        self._parse = parse_instance
        if not _IS_WINDOWS:
            return
        self._query = wintypes.HANDLE()
        self._counter = wintypes.HANDLE()
        if _PDH.PdhOpenQueryW(None, 0, ctypes.byref(self._query)) != _ERROR_SUCCESS:
            return
        if _PDH.PdhAddEnglishCounterW(self._query, counter_path, 0, ctypes.byref(self._counter)) != _ERROR_SUCCESS:
            _PDH.PdhCloseQuery(self._query)
            return
        # Most "percentage" PDH counters require two collects to differentiate.
        _PDH.PdhCollectQueryData(self._query)
        self.available = True

    def close(self) -> None:
        if _IS_WINDOWS and self.available:
            _PDH.PdhCloseQuery(self._query)
            self.available = False

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def sample(self) -> dict[int, float]:
        if not _IS_WINDOWS or not self.available:
            return {}
        if _PDH.PdhCollectQueryData(self._query) != _ERROR_SUCCESS:
            return {}
        buf_size = wintypes.DWORD(0)
        item_count = wintypes.DWORD(0)
        rc = _PDH.PdhGetFormattedCounterArrayW(
            self._counter, _PDH_FMT_DOUBLE,
            ctypes.byref(buf_size), ctypes.byref(item_count), None,
        )
        if rc & 0xFFFFFFFF != _PDH_MORE_DATA or buf_size.value == 0:
            return {}
        buf = (ctypes.c_byte * buf_size.value)()
        rc = _PDH.PdhGetFormattedCounterArrayW(
            self._counter, _PDH_FMT_DOUBLE,
            ctypes.byref(buf_size), ctypes.byref(item_count),
            ctypes.cast(buf, ctypes.c_void_p),
        )
        if rc != _ERROR_SUCCESS:
            return {}
        items = ctypes.cast(buf, ctypes.POINTER(_PDH_FMT_COUNTERVALUE_ITEM_W))
        out: dict[int, float] = {}
        for i in range(item_count.value):
            name = items[i].szName
            if not name:
                continue
            key = self._parse(name)
            if key is None:
                continue
            val = items[i].FmtValue.doubleValue
            if val:
                out[key] = out.get(key, 0.0) + float(val)
        return out


def _gpu_engine_pid(name: str):
    m = _PID_RE.match(name)
    return int(m.group(1)) if m else None


def _proc_v2_pid(name: str):
    m = _PROC_V2_RE.search(name)
    return int(m.group(1)) if m else None


class GpuEngineSampler(_PdhWildcardCounter):
    """``\\GPU Engine(*)\\Utilization Percentage`` summed by PID."""

    def __init__(self) -> None:
        super().__init__(r"\GPU Engine(*)\Utilization Percentage", _gpu_engine_pid)


class ProcessCpuSampler(_PdhWildcardCounter):
    """``\\Process V2(*)\\% Processor Time`` summed by PID.

    This bypasses :py:func:`psutil.Process.cpu_percent`, which on Windows
    relies on ``GetProcessTimes`` and frequently reports 0% for CPU-bound
    child processes because the kernel updates per-process CPU accounting
    very lazily. PDH performance counters are the same source PerfMon /
    Task Manager use, and give accurate per-PID utilization.
    """

    def __init__(self) -> None:
        super().__init__(r"\Process V2(*)\% Processor Time", _proc_v2_pid)


# --------------------------------------------------------------------------- #
# Process tree sampler                                                        #
# --------------------------------------------------------------------------- #

@dataclass
class Samples:
    """All percent values use the *system-fraction* convention.

    cpu_pct
        Percentage of total system CPU used by the process tree.
        100% means every logical core is fully busy. PDH ``\\Process V2``
        natively reports this. The psutil fallback (non-Windows) is
        normalized by dividing the per-process sum by ``cpu_count()``.
    xpu_pct
        Sum across all GPU engines for the process tree. With multiple
        engines (3D / Compute / Copy / …) this CAN exceed 100%.
    """

    t: List[float] = field(default_factory=list)
    rss_mb: List[float] = field(default_factory=list)
    cpu_pct: List[float] = field(default_factory=list)
    xpu_pct: List[float] = field(default_factory=list)
    n_procs: List[int] = field(default_factory=list)


@dataclass
class Summary:
    wall_s: float = 0.0
    peak_rss_mb: float = 0.0
    mean_rss_mb: float = 0.0
    peak_cpu_pct: float = 0.0
    mean_cpu_pct: float = 0.0
    peak_xpu_pct: float = 0.0
    mean_xpu_pct: float = 0.0
    n_samples: int = 0
    cpu_count: int = field(default_factory=lambda: psutil.cpu_count() or 1)


class _Sampler(threading.Thread):
    def __init__(
        self,
        root_proc: psutil.Process,
        gpu: GpuEngineSampler,
        cpu_pdh: Optional[ProcessCpuSampler],
        interval: float,
    ) -> None:
        super().__init__(daemon=True)
        self.root = root_proc
        self.gpu = gpu
        self.cpu_pdh = cpu_pdh   # may be None on non-Windows / unsupported builds
        self.interval = max(0.05, float(interval))
        self.samples = Samples()
        # NOTE: ``threading.Thread`` has an internal ``_stop`` method that the
        # interpreter calls during ``join``; never shadow it.
        self._stop_event = threading.Event()
        self._primed: set[int] = set()
        self._t0 = time.perf_counter()

    def _walk_tree(self) -> List[psutil.Process]:
        try:
            descendants = self.root.children(recursive=True)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            descendants = []
        return [self.root] + descendants

    def _prime(self, procs: Iterable[psutil.Process]) -> None:
        for p in procs:
            if p.pid in self._primed:
                continue
            try:
                p.cpu_percent(interval=None)
                self._primed.add(p.pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

    def run(self) -> None:
        # First pass primes the cpu_percent deltas without recording values.
        self._prime(self._walk_tree())
        next_tick = time.perf_counter() + self.interval
        while not self._stop_event.is_set():
            sleep_for = next_tick - time.perf_counter()
            if sleep_for > 0:
                if self._stop_event.wait(sleep_for):
                    break
            next_tick += self.interval

            procs = self._walk_tree()
            self._prime(procs)

            psutil_cpu = 0.0
            rss = 0
            alive = 0
            pids: set[int] = set()
            for p in procs:
                try:
                    psutil_cpu += p.cpu_percent(interval=None)
                    rss += p.memory_info().rss
                    pids.add(p.pid)
                    alive += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            # CPU%: PDH \Process V2 is accurate on Windows and natively
            # uses the system-fraction convention (max 100%). The psutil
            # fallback is per-core summed (0..N*100), so we divide by
            # cpu_count to match. If cpu_count is None just use the raw value.
            if self.cpu_pdh is not None and self.cpu_pdh.available:
                cpu_samples = self.cpu_pdh.sample()
                cpu = sum(v for pid, v in cpu_samples.items() if pid in pids)
            else:
                n_cores = psutil.cpu_count() or 1
                cpu = psutil_cpu / n_cores

            gpu_samples = self.gpu.sample() if self.gpu.available else {}
            xpu = sum(v for pid, v in gpu_samples.items() if pid in pids)

            self.samples.t.append(time.perf_counter() - self._t0)
            self.samples.rss_mb.append(rss / (1024 * 1024))
            self.samples.cpu_pct.append(cpu)
            self.samples.xpu_pct.append(xpu)
            self.samples.n_procs.append(alive)

    def stop(self) -> None:
        self._stop_event.set()


def summarize(samples: Samples, wall_s: float) -> Summary:
    s = Summary(wall_s=wall_s, n_samples=len(samples.t))
    if samples.rss_mb:
        s.peak_rss_mb = max(samples.rss_mb)
        s.mean_rss_mb = sum(samples.rss_mb) / len(samples.rss_mb)
    if samples.cpu_pct:
        s.peak_cpu_pct = max(samples.cpu_pct)
        s.mean_cpu_pct = sum(samples.cpu_pct) / len(samples.cpu_pct)
    if samples.xpu_pct:
        s.peak_xpu_pct = max(samples.xpu_pct)
        s.mean_xpu_pct = sum(samples.xpu_pct) / len(samples.xpu_pct)
    return s


# --------------------------------------------------------------------------- #
# Public entry point                                                          #
# --------------------------------------------------------------------------- #

def run_with_metrics(
    cmd: Sequence[str] | str,
    cwd: Optional[str] = None,
    env: Optional[dict] = None,
    interval: float = 0.1,
    stream: bool = True,
    print_summary: bool = True,
    label: Optional[str] = None,
) -> tuple[int, str, Samples, Summary]:
    """Run a subprocess, sample resource usage, and capture stdout.

    Parameters
    ----------
    cmd
        Command to execute. Strings are executed with ``shell=True`` (Windows
        friendly), sequences with ``shell=False`` (no quoting headaches).
    interval
        Sampling interval in seconds. 0.5s keeps overhead low while giving
        usable resolution for typical preprocess/training jobs.
    stream
        If True, mirror the child's stdout to this process's stdout while it
        runs so the notebook still shows progress lines.

    Returns
    -------
    (returncode, full_stdout_text, samples, summary)
    """
    if isinstance(cmd, str):
        args: object = cmd
        shell = True
        shown = cmd
    else:
        args = [str(a) for a in cmd]
        shell = False
        shown = " ".join(args)

    header = f"$ {shown}"
    if label:
        header = f"[{label}] {header}"
    if cwd:
        header += f"\n  (cwd={cwd})"
    print(header)

    gpu = GpuEngineSampler()
    cpu_pdh = ProcessCpuSampler()
    xpu_was_available = gpu.available
    cpu_pdh_was_available = cpu_pdh.available
    if print_summary:
        cpu_src = (
            "PDH \\Process V2(*)\\% Processor Time"
            if cpu_pdh.available
            else "psutil.Process.cpu_percent (may underreport on Windows)"
        )
        xpu_src = (
            "PDH \\GPU Engine(*)\\Utilization Percentage"
            if gpu.available
            else "unavailable"
        )
        print(f"  CPU sampler: {cpu_src}")
        print(f"  XPU sampler: {xpu_src}")

    proc = subprocess.Popen(
        args, cwd=cwd, shell=shell,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        bufsize=1, encoding="utf-8", errors="replace", env=env,
    )
    assert proc.stdout is not None

    psutil_proc = psutil.Process(proc.pid)
    sampler = _Sampler(psutil_proc, gpu, cpu_pdh, interval)
    sampler.start()

    t0 = time.perf_counter()
    captured: list[str] = []
    try:
        for line in proc.stdout:
            captured.append(line)
            if stream:
                sys.stdout.write(line)
                sys.stdout.flush()
        proc.wait()
    finally:
        sampler.stop()
        sampler.join(timeout=2.0)
        gpu.close()
        cpu_pdh.close()

    wall = time.perf_counter() - t0
    summary = summarize(sampler.samples, wall)

    if print_summary:
        xpu_note = "" if xpu_was_available else "  (XPU sampler unavailable)"
        print(
            f"\n[metrics] wall={summary.wall_s:7.3f}s  "
            f"peak_rss={summary.peak_rss_mb:7.1f}MB  "
            f"mean_cpu={summary.mean_cpu_pct:6.1f}% (peak {summary.peak_cpu_pct:.1f}%)  "
            f"mean_xpu={summary.mean_xpu_pct:5.1f}% (peak {summary.peak_xpu_pct:.1f}%){xpu_note}"
        )

    return proc.returncode, "".join(captured), sampler.samples, summary
