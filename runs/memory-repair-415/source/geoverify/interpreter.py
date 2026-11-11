"""Preserve the venv entry-point path: resolving its symlink loses pyvenv.cfg."""
import os
from pathlib import Path


def interpreter_path(path):
    return Path(os.path.abspath(os.fspath(path)))
