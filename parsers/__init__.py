#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Development parsers for Crucible datasets.

Install this package in editable mode to make parsers available to nano-crucible:

    pip install -e .

Then register each parser as an entry point in pyproject.toml:

    [project.entry-points."crucible.parsers"]
    myparser = "parsers.mymodule:MyParserClass"
"""

from .base import BaseParser

__all__ = ["BaseParser"]
