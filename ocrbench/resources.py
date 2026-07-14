"""Background resource sampler.

Used to measure peak CPU%, RAM, GPU utilization and VRAM *during* PaddleOCR
runs. psutil and pynvml are imported lazily; if unavailable, the corresponding
metrics come back as ``None`` rather than raising.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class ResourceSample:
    peak_cpu_percent: Optional[float] = None
    peak_ram_mb: Optional[float] = None
    peak_gpu_util_percent: Optional[float] = None
    peak_vram_mb: Optional[float] = None
    samples: int = 0

    def as_dict(self) -> Dict[str, Optional[float]]:
        return {
            "peak_cpu_percent": self.peak_cpu_percent,
            "peak_ram_mb": self.peak_ram_mb,
            "peak_gpu_util_percent": self.peak_gpu_util_percent,
            "peak_vram_mb": self.peak_vram_mb,
            "samples": self.samples,
        }


class ResourceSampler:
    """Poll system/GPU utilization on a background thread between start/stop."""

    def __init__(self, interval: float = 0.1, gpu: bool = False):
        self.interval = interval
        self.gpu = gpu
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._result = ResourceSample()

        self._psutil = None
        self._proc = None
        self._nvml = None
        self._gpu_handle = None

    # -- lifecycle --------------------------------------------------------
    def __enter__(self) -> "ResourceSampler":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def start(self) -> None:
        self._init_backends()
        self._stop.clear()
        self._result = ResourceSample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> ResourceSample:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval * 5))
            self._thread = None
        self._shutdown_gpu()
        return self._result

    # -- backends ---------------------------------------------------------
    def _init_backends(self) -> None:
        try:
            import psutil

            self._psutil = psutil
            self._proc = psutil.Process()
            # Prime cpu_percent so the first real reading is meaningful.
            self._proc.cpu_percent(interval=None)
            psutil.cpu_percent(interval=None)
        except Exception:
            self._psutil = None

        if self.gpu:
            try:
                import pynvml

                pynvml.nvmlInit()
                self._nvml = pynvml
                self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception:
                self._nvml = None
                self._gpu_handle = None

    def _shutdown_gpu(self) -> None:
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml = None
            self._gpu_handle = None

    # -- sampling loop ----------------------------------------------------
    def _run(self) -> None:
        # Take at least one sample even for very short runs.
        while True:
            self._sample_once()
            if self._stop.wait(self.interval):
                break
        self._sample_once()

    def _sample_once(self) -> None:
        r = self._result
        r.samples += 1

        if self._psutil is not None:
            try:
                cpu = self._psutil.cpu_percent(interval=None)
                r.peak_cpu_percent = _peak(r.peak_cpu_percent, cpu)
                ram_mb = self._proc.memory_info().rss / (1024 * 1024)
                r.peak_ram_mb = _peak(r.peak_ram_mb, ram_mb)
            except Exception:
                pass

        if self._nvml is not None and self._gpu_handle is not None:
            try:
                util = self._nvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
                r.peak_gpu_util_percent = _peak(
                    r.peak_gpu_util_percent, float(util.gpu)
                )
                mem = self._nvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                vram_mb = mem.used / (1024 * 1024)
                r.peak_vram_mb = _peak(r.peak_vram_mb, vram_mb)
            except Exception:
                pass


def _peak(current: Optional[float], new: float) -> float:
    return new if current is None else max(current, new)
