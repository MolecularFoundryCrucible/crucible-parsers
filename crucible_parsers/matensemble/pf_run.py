#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase field run parser for individual MatEnsemble MOOSE simulation directories.

Each run directory contains:
- ``out_*.csv``  — time-series postprocessing data (energies, polarization, strain …)
- ``out_*.e``    — Exodus mesh output
- ``stdout``     — MOOSE console log (commandline args, parallelism info)
- ``stderr``     — Scheduler/system log

Parameter values for this run are extracted from ``stdout`` (where MOOSE echoes
all commandline overrides) so nothing is hardcoded or assumed from the folder name.
A thumbnail is generated from the polarization time series when available.
"""

import csv
import logging
import os
import re
import tempfile
from pathlib import Path

from crucible.parsers import BaseParser

logger = logging.getLogger(__name__)


class PhaseFieldRunParser(BaseParser):
    """Parser for a single MatEnsemble phase field run directory."""

    _measurement     = "PhaseField-run"
    _data_format     = "MOOSE"
    _instrument_name = None

    # ------------------------------------------------------------------ parse

    def parse(self):
        """
        Parse a phase field run directory.

        Reads MOOSE commandline arguments from ``stdout``, extracts time-series
        metadata from the CSV output, and generates a polarization thumbnail.
        Output files (CSV, Exodus ``.e``) are too large to upload — their\n        absolute paths are recorded in metadata instead.
        """
        if not self.files_to_upload:
            raise ValueError("No input files provided")

        first   = Path(self.files_to_upload[0])
        run_dir = first if first.is_dir() else first.parent

        logger.debug(f"Parsing PhaseField run dataset: {run_dir}")

        # 1. Extract MOOSE commandline overrides and runtime info from stdout
        moose_meta = self._parse_stdout(run_dir)

        # 2. Parse CSV for time-series metadata
        csv_file = next(run_dir.glob("*.csv"), None)
        csv_meta = self._parse_csv(csv_file) if csv_file else {}

        # 3. CSV is uploaded; Exodus .e is too large — track its path in metadata
        exodus = next(run_dir.glob("*.e"), None)
        self.files_to_upload = [str(csv_file)] if csv_file else []

        # 4. Assemble metadata
        self.add_metadata({
            "root":        str(run_dir.resolve()),
            "folder_name": run_dir.name,
            "simulator":   "MOOSE",
            "csv_file":    str(csv_file) if csv_file else None,
            "exodus_file": str(exodus) if exodus else None,
            **moose_meta,
            **csv_meta,
        })

        # 5. Keywords from parameter values
        kws = ["phase field", "MOOSE", "matensemble", "phase-field-run"]
        for val in moose_meta.get("moose_args", {}).values():
            kws.append(str(val).lower())
        self.add_keywords(kws)

        # 6. Thumbnail: polarization vs time from CSV
        if csv_file and "avg_pz" in csv_meta.get("columns", []):
            self.thumbnail = self._plot_polarization(csv_file, self.mfid)

    # --------------------------------------------------------- static helpers

    @staticmethod
    def _parse_stdout(run_dir: Path) -> dict:
        """
        Extract MOOSE metadata from the ``stdout`` log.

        Parses:
        - ``Command Line Input Argument(s)`` block → parameter key=value pairs
        - ``Input File(s)`` → MOOSE input file path
        - ``Num Processors`` → parallelism
        - ``Current Time`` → wall-clock timestamp
        """
        stdout = run_dir / "stdout"
        if not stdout.exists():
            logger.warning(f"No stdout found in {run_dir}")
            return {}

        text = stdout.read_text(errors="replace")
        meta = {}
        args = {}

        lines = text.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i]

            if "Command Line Input Argument(s):" in line:
                i += 1
                while i < len(lines):
                    arg_line = lines[i].strip()
                    if not arg_line or (":" in arg_line and "=" not in arg_line):
                        break
                    if "=" in arg_line:
                        key, _, val = arg_line.partition("=")
                        short_key = key.strip().rsplit("/", 1)[-1]
                        val = val.strip()
                        try:
                            val = float(val)
                        except ValueError:
                            pass
                        args[short_key] = val
                    i += 1
                continue

            if "Input File(s):" in line and i + 1 < len(lines):
                meta["moose_input_file"] = lines[i + 1].strip()

            m = re.search(r"Num Processors:\s*(\d+)", line)
            if m:
                meta["num_processors"] = int(m.group(1))

            m = re.search(r"Current Time:\s*(.+)", line)
            if m:
                meta["run_timestamp"] = m.group(1).strip()

            i += 1

        if args:
            meta["moose_args"] = args
        return meta

    @staticmethod
    def _parse_csv(csv_file: Path) -> dict:
        """
        Read a MOOSE postprocessor CSV for time-series metadata.

        Returns column names, number of timesteps, and start/end times.
        """
        try:
            with open(csv_file, newline="") as fh:
                reader = csv.DictReader(fh)
                rows = list(reader)
            if not rows:
                return {}
            return {
                "columns":     list(rows[0].keys()),
                "n_timesteps": len(rows),
                "time_start":  float(rows[0].get("time", 0)),
                "time_end":    float(rows[-1].get("time", 0)),
            }
        except Exception as exc:
            logger.warning(f"Could not parse CSV {csv_file}: {exc}")
            return {}

    @staticmethod
    def _plot_polarization(csv_file: Path, mfid: str | None) -> str | None:
        """
        Generate a thumbnail plot of average polarization (``avg_pz``) vs time.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            times, pz = [], []
            with open(csv_file, newline="") as fh:
                for row in csv.DictReader(fh):
                    try:
                        times.append(float(row["time"]))
                        pz.append(float(row["avg_pz"]))
                    except (KeyError, ValueError):
                        pass

            if not times:
                return None

            fig, ax = plt.subplots(figsize=(4, 3))
            ax.plot(times, pz, lw=1.5)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel(r"$\langle P_z \rangle$ (C m$^{-2}$)")
            ax.set_title("Polarization vs Time")
            fig.tight_layout()

            thumb_dir = os.path.join(tempfile.gettempdir(), "crucible_thumbnails")
            os.makedirs(thumb_dir, exist_ok=True)
            fname = f"{mfid}.png" if mfid else f"{csv_file.stem}_thumbnail.png"
            path = os.path.join(thumb_dir, fname)
            fig.savefig(path, dpi=100)
            plt.close(fig)
            return path

        except ImportError:
            logger.warning("matplotlib not installed; skipping thumbnail generation")
        except Exception as exc:
            logger.warning(f"Could not generate thumbnail from {csv_file}: {exc}")
        return None
