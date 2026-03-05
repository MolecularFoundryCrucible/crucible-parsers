#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase field manager parser for the top-level simulation setup dataset.

Parses the root directory of a MatEnsemble phase field campaign, which holds
the shared input files (MOOSE input script, driver script) that all individual
runs share.  Metadata is extracted from the driver script; nothing about the
material system is hardcoded.
"""

import ast
import logging
import re
from pathlib import Path

from crucible.parsers import BaseParser

logger = logging.getLogger(__name__)


class PhaseFieldManagerParser(BaseParser):
    """Parser for the MatEnsemble phase field root/manager dataset."""

    _measurement     = "PhaseField-manager"
    _data_format     = "MOOSE"
    _instrument_name = None

    # ------------------------------------------------------------------ parse

    def parse(self):
        """
        Parse the phase field root simulation directory.

        Accepts either the root directory or the driver Python script as input.
        Extracts parameter sweep definition and MOOSE input file reference from
        the driver script.  All non-hidden regular files in the root directory
        are uploaded.
        """
        if not self.files_to_upload:
            raise ValueError("No input files provided")

        first = Path(self.files_to_upload[0])
        if first.suffix == ".py":
            driver   = first
            root_dir = first.parent
        else:
            root_dir = first if first.is_dir() else first.parent
            py_files = sorted(root_dir.glob("*.py"))
            driver   = py_files[0] if py_files else None

        logger.debug(f"Parsing PhaseField manager dataset: {root_dir}")

        driver_text = driver.read_text(errors="replace") if driver else ""

        # Locate MOOSE input file referenced in the driver script
        moose_input = self._find_moose_input(driver_text, root_dir)

        # Extract parameter sweep definition from driver script
        params = self._extract_params(driver_text)

        self.add_metadata({
            "root":        str(root_dir.resolve()),
            "simulator":   "MOOSE",
            "base_input":  str(moose_input) if moose_input else None,
            **params,
        })

        # Upload driver script + MOOSE input file
        files = []
        if driver:
            files.append(str(driver))
        if moose_input and moose_input.exists():
            files.append(str(moose_input))
        self.files_to_upload = files

        kws = ["phase field", "MOOSE", "matensemble", "phase-field-manager"]
        if moose_input:
            kws.append(moose_input.stem.lower())
        self.add_keywords(kws)

    # --------------------------------------------------------- static helpers

    @staticmethod
    def _find_moose_input(text: str, root_dir: Path) -> Path | None:
        """Find the MOOSE ``.i`` input file referenced in the driver script."""
        # Match quoted .i filenames
        m = re.search(r"['\"]([^'\"]+\.i)['\"]", text)
        if m:
            candidate = root_dir / m.group(1)
            if candidate.exists():
                return candidate
        # Fallback: first .i file in root
        candidates = sorted(root_dir.glob("*.i"))
        return candidates[0] if candidates else None

    @staticmethod
    def _extract_params(text: str) -> dict:
        """
        Extract parameter arrays and key settings from the driver script.

        Looks for assignments of the form ``name_values = [...]`` or
        ``name_values = np.asarray([...])``, plus ``num_cores`` and the
        MOOSE app path.
        """
        params = {}

        for m in re.finditer(
            r"(\w+_values)\s*=\s*(?:np\.asarray\()?(\[.*?\])(?:\))?",
            text, re.DOTALL
        ):
            key = m.group(1)
            raw = m.group(2).strip()
            try:
                params[key] = ast.literal_eval(raw)
            except Exception:
                params[key] = raw

        m = re.search(r"['\"]num_cores['\"]\s*:\s*(\d+)", text)
        if m:
            params["num_cores"] = int(m.group(1))

        m = re.search(r"mooseapp\s*=\s*['\"]([^'\"]+)['\"]", text)
        if m:
            params["moose_app"] = m.group(1)

        return params
