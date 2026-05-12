"""Load QDF bench ``sampler`` (PDH CPU + GPU engine %) for in-process measurement.

Falls back to a lightweight psutil-only sampler if the QDF tree is missing.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional

import psutil

REPO = Path(__file__).resolve().parents[3]
QDF_SAMPLER = REPO / "QuantumDeepField_molecule" / "bench" / "sampler.py"


@dataclass
class _FallbackSamples:
    t: List[float] = field(default_factory=list)
    rss_mb: List[float] = field(default_factory=list)
    cpu_pct: List[float] = field(default_factory=list)
    xpu_pct: List[float] = field(default_factory=list)
    n_procs: List[int] = field(default_factory=list)


class _FallbackSampler(threading.Thread):
    def __init__(self, interval: float) -> None:
        super().__init__(daemon=True)
        self.interval = max(0.05, float(interval))
        self.samples = _FallbackSamples()
        self._stop = threading.Event()
        self._t0 = time.perf_counter()
        self._proc = psutil.Process(os.getpid())

    def run(self) -> None:
        self._proc.cpu_percent(interval=None)
        next_tick = time.perf_counter() + self.interval
        while not self._stop.is_set():
            sleep_for = next_tick - time.perf_counter()
            if sleep_for > 0:
                if self._stop.wait(sleep_for):
                    break
            next_tick += self.interval
            try:
                cpu = self._proc.cpu_percent(interval=None)
                rss = self._proc.memory_info().rss
                n_cores = psutil.cpu_count() or 1
                self.samples.t.append(time.perf_counter() - self._t0)
                self.samples.rss_mb.append(rss / (1024 * 1024))
                self.samples.cpu_pct.append(cpu / n_cores)
                self.samples.xpu_pct.append(0.0)
                self.samples.n_procs.append(1)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break

    def stop(self) -> None:
        self._stop.set()


def _load_qdf_sampler_module():
    if not QDF_SAMPLER.is_file():
        return None
    spec = importlib.util.spec_from_file_location("qdf_bench_sampler", QDF_SAMPLER)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def measure_callable(
    fn: Callable[[], Any],
    *,
    interval: float = 0.1,
) -> tuple[Any, Any, Any]:
    """Run ``fn()`` while background-sampling CPU/XPU/RSS.

    Returns ``(fn_result, samples, summary)`` where ``samples`` matches QDF
    :class:`Samples` layout (``.t``, ``.cpu_pct``, ``.xpu_pct``, ``.rss_mb``).
    """
    mod = _load_qdf_sampler_module()
    if mod is not None:
        root = psutil.Process(os.getpid())
        gpu = mod.GpuEngineSampler()
        cpu_pdh = mod.ProcessCpuSampler()
        sampler = mod._Sampler(root, gpu, cpu_pdh, interval)
        sampler.start()
        t0 = time.perf_counter()
        try:
            out = fn()
        finally:
            sampler.stop()
            sampler.join(timeout=5.0)
            gpu.close()
            cpu_pdh.close()
        wall = time.perf_counter() - t0
        summary = mod.summarize(sampler.samples, wall)
        return out, sampler.samples, summary

    fb = _FallbackSampler(interval)
    fb.start()
    t0 = time.perf_counter()
    try:
        out = fn()
    finally:
        fb.stop()
        fb.join(timeout=5.0)
    wall = time.perf_counter() - t0
    s = fb.samples

    class _S:
        wall_s = wall
        n_samples = len(s.t)
        peak_rss_mb = max(s.rss_mb) if s.rss_mb else 0.0
        mean_rss_mb = sum(s.rss_mb) / len(s.rss_mb) if s.rss_mb else 0.0
        peak_cpu_pct = max(s.cpu_pct) if s.cpu_pct else 0.0
        mean_cpu_pct = sum(s.cpu_pct) / len(s.cpu_pct) if s.cpu_pct else 0.0
        peak_xpu_pct = 0.0
        mean_xpu_pct = 0.0

    summary = _S()
    return out, s, summary
