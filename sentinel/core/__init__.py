"""Sentinel Trader — core orchestration: scheduler, pipeline, position manager."""

from __future__ import annotations

from sentinel.core.daily_report import run_daily_report
from sentinel.core.pipeline import PipelineContext, run_analysis_pipeline
from sentinel.core.positions import manage_open_positions
from sentinel.core.scheduler import run_scan_scheduler
from sentinel.core.watchdog import run_watchdog

__all__ = [
    "PipelineContext",
    "manage_open_positions",
    "run_analysis_pipeline",
    "run_daily_report",
    "run_scan_scheduler",
    "run_watchdog",
]

